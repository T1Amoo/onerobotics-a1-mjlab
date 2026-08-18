"""Compare the packaged A1 model with the canonical OneRobotics source model."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np

mj = cast(Any, mujoco)

JOINT_NAMES = [f"joint{i}-a1_r" for i in range(1, 8)]
BODY_NAMES = [f"Link{i}" for i in range(1, 8)]
MESH_NAMES = ["base_link.STL", *[f"Link_R{i}.STL" for i in range(1, 8)]]

JOINT_FIELDS = (
  "jnt_type",
  "jnt_pos",
  "jnt_axis",
  "jnt_range",
  "jnt_stiffness",
  "jnt_solref",
  "jnt_solimp",
)
DOF_FIELDS = ("dof_armature", "dof_damping", "dof_frictionloss")
BODY_FIELDS = (
  "body_pos",
  "body_quat",
  "body_mass",
  "body_inertia",
  "body_ipos",
  "body_iquat",
)
GEOM_FIELDS = (
  "geom_type",
  "geom_contype",
  "geom_conaffinity",
  "geom_condim",
  "geom_priority",
  "geom_friction",
  "geom_solref",
  "geom_solimp",
  "geom_size",
  "geom_pos",
  "geom_quat",
  "geom_rgba",
)
ACTUATOR_FIELDS = (
  "actuator_trntype",
  "actuator_dyntype",
  "actuator_gaintype",
  "actuator_biastype",
  "actuator_trnid",
  "actuator_gear",
  "actuator_gainprm",
  "actuator_biasprm",
  "actuator_ctrlrange",
  "actuator_forcerange",
  "actuator_forcelimited",
  "actuator_ctrllimited",
)


def _array(value: Any, indices: list[int] | None = None) -> list[Any]:
  array = np.asarray(value)
  if indices is not None:
    array = array[indices]
  if np.issubdtype(array.dtype, np.floating):
    array = np.round(array.astype(np.float64), decimals=12)
  return array.tolist()


def _mesh_dir(xml_path: Path) -> Path:
  compiler = ET.parse(xml_path).getroot().find("compiler")
  if compiler is None or "meshdir" not in compiler.attrib:
    raise ValueError(f"No compiler meshdir in {xml_path}")
  return (xml_path.parent / compiler.attrib["meshdir"]).resolve()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def extract_groups(xml_path: Path, *, canonical_source: bool) -> dict[str, Any]:
  """Extract the 55 named fidelity groups from a compiled A1 model."""
  model = mj.MjSpec.from_file(str(xml_path.resolve())).compile()
  joint_ids = [model.joint(name).id for name in JOINT_NAMES]
  dof_ids = [model.jnt_dofadr[joint_id] for joint_id in joint_ids]
  body_ids = [model.body(name).id for name in BODY_NAMES]

  # The source contains two base geoms, a demo floor, fourteen link geoms, and
  # a demo goal. The standalone model contains only the same sixteen robot
  # geoms, in unchanged order, with descriptive names.
  geom_ids = [0, 1, *range(3, 17)] if canonical_source else list(range(16))
  if len(geom_ids) != 16:
    raise AssertionError("Expected sixteen A1 robot geoms")

  groups: dict[str, Any] = {}
  for field in JOINT_FIELDS:
    groups[field] = _array(getattr(model, field), joint_ids)
  for field in DOF_FIELDS:
    groups[field] = _array(getattr(model, field), dof_ids)
  for field in BODY_FIELDS:
    groups[field] = _array(getattr(model, field), body_ids)
  for field in GEOM_FIELDS:
    groups[field] = _array(getattr(model, field), geom_ids)
  for field in ACTUATOR_FIELDS:
    groups[field] = _array(getattr(model, field))

  key_name = "home"
  key = model.key(key_name)
  groups["home_qpos"] = _array(key.qpos)
  groups["home_ctrl"] = _array(key.ctrl)
  groups["joint_names"] = [model.joint(index).name for index in joint_ids]
  groups["body_names"] = [model.body(index).name for index in body_ids]
  groups["actuator_names"] = [model.actuator(index).name for index in range(model.nu)]
  groups["option_timestep"] = round(float(model.opt.timestep), 12)
  groups["option_integrator"] = int(model.opt.integrator)

  mesh_dir = _mesh_dir(xml_path)
  for mesh_name in MESH_NAMES:
    mesh_path = mesh_dir / mesh_name
    if not mesh_path.is_file():
      raise FileNotFoundError(mesh_path)
    groups[f"mesh_sha256:{mesh_name}"] = _sha256(mesh_path)

  if len(groups) != 55:
    raise AssertionError(f"Expected 55 fidelity groups, found {len(groups)}")
  return groups


def values_match(actual: Any, expected: Any) -> bool:
  """Compare a fidelity group with strict names and tolerant floating values."""
  try:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.dtype.kind in "f" or expected_array.dtype.kind in "f":
      return bool(
        np.allclose(
          actual_array.astype(np.float64),
          expected_array.astype(np.float64),
          rtol=1.0e-10,
          atol=1.0e-12,
        )
      )
  except (TypeError, ValueError):
    pass
  return actual == expected


def compare_groups(
  candidate: dict[str, Any],
  reference: dict[str, Any],
) -> list[str]:
  """Return the names of fidelity groups that differ."""
  if set(candidate) != set(reference):
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    return [f"group-set missing={missing} extra={extra}"]
  return [
    name
    for name in sorted(reference)
    if not values_match(candidate[name], reference[name])
  ]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument(
    "--write-reference",
    type=Path,
    help="Write a JSON reference extracted from --source.",
  )
  args = parser.parse_args()

  reference = extract_groups(args.source, canonical_source=True)
  if args.write_reference is not None:
    payload = {
      "schema": 1,
      "source": (
        "katazen/onerobot_h1@ca6d705f37b0dc296bfe7f33f7c83d780c3d3a70 "
        "canonical a1_right_position.xml"
      ),
      "groups": reference,
    }
    args.write_reference.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )

  candidate = extract_groups(args.candidate, canonical_source=False)
  mismatches = compare_groups(candidate, reference)
  passed = len(reference) - len(mismatches)
  print(f"Physics fidelity: {passed} / {len(reference)} PASS")
  if mismatches:
    print("Differences:")
    for name in mismatches:
      print(f"  - {name}")
    return 1
  print("Physics/model parameter differences from canonical source: NONE")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
