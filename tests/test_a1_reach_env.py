"""Registration and runtime checks for the A1 reach environment."""

from typing import Any, cast

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from onerobotics_a1_mjlab import TASK_ID
from onerobotics_a1_mjlab.a1 import A1_END_EFFECTOR_SITE, get_spec
from onerobotics_a1_mjlab.reach import a1_reach_env_cfg
from onerobotics_a1_mjlab.reach.mdp.actions import RateLimitedJointPositionAction
from onerobotics_a1_mjlab.reach.mdp.commands import ReachablePoseCommand

mj = cast(Any, mujoco)


def _assert_finite_observations(
  observations: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> None:
  for value in observations.values():
    if isinstance(value, dict):
      assert all(torch.isfinite(item).all() for item in value.values())
    else:
      assert torch.isfinite(value).all()


def _make_env(num_envs: int = 2) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = num_envs
  cfg.commands["ee_pose"].debug_vis = False
  return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def test_task_registration_and_configs() -> None:
  assert TASK_ID == "Mjlab-Reach-OneRobotics-A1"
  assert TASK_ID in list_tasks()
  cfg = load_env_cfg(TASK_ID)
  play_cfg = load_env_cfg(TASK_ID, play=True)
  rl_cfg = load_rl_cfg(TASK_ID)
  direct_cfg = a1_reach_env_cfg()
  other_direct_cfg = a1_reach_env_cfg()
  actor_joint_pos = direct_cfg.observations["actor"].terms["joint_pos"]
  critic_joint_pos = direct_cfg.observations["critic"].terms["joint_pos"]
  assert actor_joint_pos is not critic_joint_pos
  assert (
    actor_joint_pos.params["asset_cfg"]
    is not (
      other_direct_cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"]
    )
  )
  assert cfg.episode_length_s == 8.0
  assert play_cfg.episode_length_s > 1.0e8
  assert rl_cfg.experiment_name == "onerobotics_a1_reach"


def test_reach_env_reset_and_zero_random_action_smoke() -> None:
  env = _make_env()
  try:
    timestep = env.sim.model.opt.timestep[:]
    torch.testing.assert_close(timestep, torch.full_like(timestep, 1.0 / 60.0))
    assert env.sim.model.opt.integrator == int(mj.mjtIntegrator.mjINT_IMPLICITFAST)

    observations, _ = env.reset()
    assert env.action_manager.total_action_dim == 7
    assert env.single_action_space.shape == (7,)
    actor = observations["actor"]
    critic = observations["critic"]
    assert isinstance(actor, torch.Tensor)
    assert isinstance(critic, torch.Tensor)
    assert actor.shape == (2, 35)
    assert critic.shape == (2, 35)
    _assert_finite_observations(observations)

    zero_actions = torch.zeros((2, 7), device=env.device)
    observations, rewards, terminated, truncated, _ = env.step(zero_actions)
    _assert_finite_observations(observations)
    assert torch.isfinite(rewards).all()
    assert not terminated.any()
    assert not truncated.any()

    action_term = env.action_manager.get_term("joint_pos")
    assert isinstance(action_term, RateLimitedJointPositionAction)
    previous_target = action_term.limited_target.clone()
    for _ in range(100):
      random_actions = 2.0 * torch.rand((2, 7), device=env.device) - 1.0
      observations, rewards, _, _, _ = env.step(random_actions)
      target_delta = torch.abs(action_term.limited_target - previous_target)
      assert torch.all(target_delta <= 0.1 + 1.0e-6)
      previous_target = action_term.limited_target.clone()
    _assert_finite_observations(observations)
    assert torch.isfinite(rewards).all()
    assert torch.isfinite(env.scene["robot"].data.joint_pos).all()
  finally:
    env.close()


def test_sampled_targets_are_reachable_by_construction() -> None:
  env = _make_env(num_envs=4)
  try:
    env.reset()
    command = env.command_manager.get_term("ee_pose")
    assert isinstance(command, ReachablePoseCommand)
    assert command.command.shape == (4, 7)
    assert command.target_joint_pos.shape == (4, 7)
    assert torch.isfinite(command.command).all()
    assert torch.isfinite(command.target_joint_pos).all()
    quat_norm = torch.linalg.vector_norm(command.command[:, 3:], dim=-1)
    torch.testing.assert_close(quat_norm, torch.ones_like(quat_norm))

    model = get_spec().compile()
    data = mj.MjData(model)
    joint_qpos_adr = np.asarray(
      [model.jnt_qposadr[model.joint(f"joint{i}-a1_r").id] for i in range(1, 8)]
    )
    site_id = model.site(A1_END_EFFECTOR_SITE).id
    center = 0.5 * (model.jnt_range[:, 0] + model.jnt_range[:, 1])
    half_range = 0.4 * (model.jnt_range[:, 1] - model.jnt_range[:, 0])

    sampled_joint_pos = command.target_joint_pos.cpu().numpy()
    sampled_pose = command.command.cpu().numpy()
    assert np.all(sampled_joint_pos >= center - half_range - 1e-6)
    assert np.all(sampled_joint_pos <= center + half_range + 1e-6)

    expected_quat = np.empty(4)
    for joint_pos, target_pose in zip(sampled_joint_pos, sampled_pose, strict=True):
      mj.mj_resetData(model, data)
      data.qpos[joint_qpos_adr] = joint_pos
      mj.mj_forward(model, data)
      mj.mju_mat2Quat(expected_quat, data.site_xmat[site_id])
      if np.dot(expected_quat, target_pose[3:]) < 0.0:
        expected_quat *= -1.0
      np.testing.assert_allclose(target_pose[:3], data.site_xpos[site_id], atol=1e-6)
      np.testing.assert_allclose(target_pose[3:], expected_quat, atol=1e-6)
  finally:
    env.close()
