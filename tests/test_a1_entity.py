"""mjlab Entity checks for the canonical OneRobotics A1."""

from typing import Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from onerobotics_a1_mjlab.a1 import A1_HOME_JOINT_POS, get_a1_robot_cfg

EXPECTED_JOINT_NAMES = [f"joint{i}-a1_r" for i in range(1, 8)]
EXPECTED_BODY_NAMES = [f"Link{i}" for i in range(1, 8)]


def test_entity_cfg_and_entity_construct(a1_entity: Entity) -> None:
  cfg = get_a1_robot_cfg()
  other_cfg = get_a1_robot_cfg()
  assert cfg.articulation is not None
  assert other_cfg.articulation is not None
  assert cfg.init_state is not other_cfg.init_state
  assert cfg.init_state.joint_pos is not other_cfg.init_state.joint_pos
  assert cfg.articulation is not other_cfg.articulation
  assert cfg.collisions[0] is not other_cfg.collisions[0]
  assert a1_entity.num_actuators == 7
  assert a1_entity.num_joints == 7
  assert a1_entity.num_bodies == 8
  assert a1_entity.is_actuated
  assert a1_entity.is_articulated
  assert a1_entity.is_fixed_base
  assert list(a1_entity.joint_names) == EXPECTED_JOINT_NAMES
  assert list(a1_entity.body_names[1:]) == EXPECTED_BODY_NAMES


def test_entity_keyframe_and_collision_configuration(a1_model: Any) -> None:
  key = a1_model.key("init_state")
  expected = np.array(list(A1_HOME_JOINT_POS.values()))
  np.testing.assert_allclose(key.qpos, expected)
  np.testing.assert_allclose(key.ctrl, expected)

  collision_geoms = [
    a1_model.geom(i)
    for i in range(a1_model.ngeom)
    if "_collision" in a1_model.geom(i).name
  ]
  assert len(collision_geoms) == 7
  for geom in collision_geoms:
    assert geom.contype == 1
    assert geom.conaffinity == 1
    assert geom.condim == 3
    assert geom.priority == 0


def test_entity_cpu_reset_and_simulation_smoke() -> None:
  entity = Entity(get_a1_robot_cfg())
  model = entity.compile()
  timestep = 1.0 / 60.0
  sim = Simulation(
    num_envs=2,
    cfg=SimulationCfg(
      njmax=256,
      mujoco=MujocoCfg(timestep=timestep, integrator="implicitfast"),
    ),
    model=model,
    device="cpu",
  )
  entity.initialize(model, sim.model, sim.data, "cpu")

  home = entity.data.default_joint_pos.clone()
  entity.write_joint_state_to_sim(home, torch.zeros_like(home))
  entity.set_joint_position_target(home)
  sim.forward()
  for _ in range(20):
    entity.write_data_to_sim()
    sim.step()
    entity.update(timestep)

  for tensor in (sim.data.qpos, sim.data.qvel, sim.data.qacc, sim.data.ctrl):
    assert torch.isfinite(tensor).all()
