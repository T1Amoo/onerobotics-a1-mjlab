"""Smoke-test an installed distribution outside the source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import mjlab
import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

mj = cast(Any, mujoco)
TASK_ID = "Mjlab-Reach-OneRobotics-A1"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workspace", type=Path, required=True)
  args = parser.parse_args()

  workspace = args.workspace.resolve()
  assert mjlab.__file__ is not None
  mjlab_path = Path(mjlab.__file__).resolve()

  # Importing mjlab above must discover the installed entry point before this
  # script imports the plugin package directly.
  assert TASK_ID in list_tasks(), "Installed mjlab.tasks entry point was not discovered"

  # Deliberately local: discovery must be checked before an explicit import can
  # register the task as a side effect.
  from onerobotics_a1_mjlab import get_a1_robot_cfg
  from onerobotics_a1_mjlab.a1 import A1_XML, get_spec

  package_xml = A1_XML.resolve()
  assert not mjlab_path.is_relative_to(workspace), mjlab_path
  assert not package_xml.is_relative_to(workspace), package_xml

  model = get_spec().compile()
  data = mj.MjData(model)
  mj.mj_resetDataKeyframe(model, data, model.key("home").id)
  for _ in range(20):
    mj.mj_step(model, data)

  entity = Entity(get_a1_robot_cfg())
  entity_model = entity.compile()
  assert (entity_model.njnt, entity_model.nu) == (7, 7)

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["ee_pose"].debug_vis = False
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  try:
    observations, _ = env.reset()
    actor = observations["actor"]
    assert isinstance(actor, torch.Tensor)
    assert actor.shape == (1, 35)
    observations, rewards, _, _, _ = env.step(torch.zeros((1, 7)))
    actor = observations["actor"]
    assert isinstance(actor, torch.Tensor)
    assert torch.isfinite(actor).all()
    assert torch.isfinite(rewards).all()
  finally:
    env.close()

  print(f"mjlab.__file__={mjlab_path}")
  print(f"onerobotics_a1_mjlab asset={package_xml}")
  print("isolated distribution smoke: PASS")


if __name__ == "__main__":
  main()
