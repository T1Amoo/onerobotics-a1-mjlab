"""Manager-based environment configuration for OneRobotics A1 pose reaching."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from onerobotics_a1_mjlab.a1 import A1_ACTION_SCALE, get_a1_robot_cfg
from onerobotics_a1_mjlab.reach import mdp


def _robot_joints_cfg() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(r"joint[1-7]-a1_r",))


def a1_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create an independent OneRobotics A1 reach environment configuration.

  Args:
    play: Disable actor observation noise and replace the 8 s training timeout
      with an effectively infinite timeout when ``True``.
  """
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"asset_cfg": _robot_joints_cfg()},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      params={"asset_cfg": _robot_joints_cfg()},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
    "target_pose": ObservationTermCfg(
      func=mdp.target_pose_b,
      params={"command_name": "ee_pose"},
    ),
    "ee_pose_error": ObservationTermCfg(
      func=mdp.ee_pose_error_b,
      params={"command_name": "ee_pose"},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=deepcopy(actor_terms),
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.RateLimitedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(r"joint[1-7]-a1_r",),
      scale=dict(A1_ACTION_SCALE),
      use_default_offset=True,
      preserve_order=True,
      max_target_delta=0.1,
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "ee_pose": mdp.ReachablePoseCommandCfg(
      resampling_time_range=(4.0, 4.0),
      joint_range_scale=0.8,
      debug_vis=True,
    ),
  }

  rewards = {
    "position_error": RewardTermCfg(
      func=mdp.position_error,
      weight=-0.5,
      params={"command_name": "ee_pose"},
    ),
    "orientation_error": RewardTermCfg(
      func=mdp.orientation_error,
      weight=-0.3,
      params={"command_name": "ee_pose"},
    ),
    "pose_tracking_coarse": RewardTermCfg(
      func=mdp.pose_tracking_tanh,
      weight=0.5,
      params={
        "command_name": "ee_pose",
        "pos_std": 0.25,
        "ori_std": 0.5,
      },
    ),
    "pose_tracking_fine": RewardTermCfg(
      func=mdp.pose_tracking_tanh,
      weight=0.5,
      params={
        "command_name": "ee_pose",
        "pos_std": 0.05,
        "ori_std": 0.1,
      },
    ),
    "action_rate_l2": RewardTermCfg(
      func=base_mdp.action_rate_l2,
      weight=-0.01,
    ),
    "joint_vel_l2": RewardTermCfg(
      func=base_mdp.joint_vel_l2,
      weight=-0.001,
      params={"asset_cfg": _robot_joints_cfg()},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": _robot_joints_cfg()},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={"robot": get_a1_robot_cfg()},
      num_envs=1,
      env_spacing=1.0,
      extent=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="Link4",
      distance=1.2,
      elevation=-15.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=32,
      njmax=128,
      mujoco=MujocoCfg(
        timestep=1.0 / 60.0,
        integrator="implicitfast",
      ),
    ),
    decimation=1,
    episode_length_s=1.0e9 if play else 8.0,
  )
