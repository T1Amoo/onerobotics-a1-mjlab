"""Smoke-test an installed distribution outside the source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import mjlab
import mujoco
from mjlab.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg

from onerobotics_a1_mjlab import TASK_ID, get_a1_robot_cfg
from onerobotics_a1_mjlab.a1 import A1_XML, get_spec

mj = cast(Any, mujoco)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workspace", type=Path, required=True)
  args = parser.parse_args()

  workspace = args.workspace.resolve()
  assert mjlab.__file__ is not None
  mjlab_path = Path(mjlab.__file__).resolve()
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
  assert TASK_ID in list_tasks()
  assert load_env_cfg(TASK_ID).actions is not None

  print(f"mjlab.__file__={mjlab_path}")
  print(f"onerobotics_a1_mjlab asset={package_xml}")
  print("isolated distribution smoke: PASS")


if __name__ == "__main__":
  main()
