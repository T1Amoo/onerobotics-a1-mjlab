"""Native MuJoCo checks for the canonical OneRobotics A1 model."""

import xml.etree.ElementTree as ET
from typing import Any, cast

import mujoco
import numpy as np

from onerobotics_a1_mjlab.a1 import (
  A1_HOME_JOINT_POS,
  A1_XML,
  get_spec,
)

mj = cast(Any, mujoco)

EXPECTED_JOINT_NAMES = [f"joint{i}-a1_r" for i in range(1, 8)]
EXPECTED_BODY_NAMES = [f"Link{i}" for i in range(1, 8)]
EXPECTED_JOINT_RANGES = np.array(
  [
    [-1.04, 3.14],
    [-3.14, 0.26],
    [-2.75, 2.75],
    [-1.92, 1.92],
    [-2.75, 2.75],
    [-1.57, 1.57],
    [-2.75, 2.75],
  ]
)


def test_mjcf_exists_loads_and_compiles() -> None:
  assert A1_XML.is_file()
  spec = get_spec()
  assert isinstance(spec, mj.MjSpec)
  model = spec.compile()
  assert model.nq == 7
  assert model.nv == 7
  assert model.njnt == 7
  assert model.nbody == 9  # world, mocap base, Link1 through Link7.
  assert model.nu == 7
  assert model.nsite == 1


def test_all_eight_mesh_paths_resolve() -> None:
  root = ET.parse(A1_XML).getroot()
  compiler = root.find("compiler")
  assert compiler is not None
  meshdir = A1_XML.parent / compiler.attrib["meshdir"]
  mesh_files = [mesh.attrib["file"] for mesh in root.findall("./asset/mesh")]
  assert len(mesh_files) == 8
  assert len(set(mesh_files)) == 8
  assert all((meshdir / mesh_file).is_file() for mesh_file in mesh_files)


def test_model_names_ranges_and_actuators() -> None:
  model = get_spec().compile()
  assert [model.joint(i).name for i in range(model.njnt)] == EXPECTED_JOINT_NAMES
  assert [model.body(i).name for i in range(2, model.nbody)] == EXPECTED_BODY_NAMES
  assert [model.actuator(i).name for i in range(model.nu)] == EXPECTED_JOINT_NAMES
  np.testing.assert_allclose(model.jnt_range, EXPECTED_JOINT_RANGES)
  assert np.all(model.jnt_limited)
  assert np.all(model.jnt_range[:, 0] < model.jnt_range[:, 1])


def test_original_xml_actuator_parameters_are_preserved() -> None:
  model = get_spec().compile()
  np.testing.assert_allclose(model.actuator_gainprm[:, 0], [60] * 4 + [30] * 3)
  np.testing.assert_allclose(model.actuator_biasprm[:, 1], [-60] * 4 + [-30] * 3)
  np.testing.assert_allclose(model.actuator_biasprm[:, 2], [-6] * 4 + [-3] * 3)
  np.testing.assert_allclose(
    model.actuator_forcerange,
    [[-30, 30]] * 4 + [[-12, 12]] * 3,
  )
  np.testing.assert_allclose(model.actuator_ctrlrange, EXPECTED_JOINT_RANGES)
  assert np.all(model.actuator_forcelimited)
  assert np.all(model.actuator_ctrllimited)


def test_home_keyframe_and_initial_state_are_finite() -> None:
  model = get_spec().compile()
  key = model.key("home")
  expected = np.array(list(A1_HOME_JOINT_POS.values()))
  np.testing.assert_allclose(key.qpos, expected)
  np.testing.assert_allclose(key.ctrl, expected)

  data = mj.MjData(model)
  mj.mj_resetDataKeyframe(model, data, key.id)
  mj.mj_forward(model, data)
  assert np.isfinite(data.qpos).all()
  assert np.isfinite(data.qvel).all()
  assert np.isfinite(data.xpos).all()


def test_native_mujoco_simulation_smoke() -> None:
  model = get_spec().compile()
  data = mj.MjData(model)
  mj.mj_resetDataKeyframe(model, data, model.key("home").id)
  for _ in range(100):
    mj.mj_step(model, data)
  for array in (data.qpos, data.qvel, data.qacc, data.ctrl, data.xpos):
    assert np.isfinite(array).all()
