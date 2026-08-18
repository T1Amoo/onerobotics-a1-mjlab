"""OneRobotics A1 model constants and mjlab entity configuration."""

from pathlib import Path
from typing import Any, cast

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_HERE = Path(__file__).parent
mj = cast(Any, mujoco)

A1_XML: Path = _HERE / "xmls" / "a1.xml"
"""Canonical position-controlled OneRobotics A1 right-arm MJCF."""

A1_END_EFFECTOR_SITE = "end_effector"
"""Site colocated with the canonical Link7 terminal frame."""

assert A1_XML.exists()


def get_spec() -> Any:
  """Load a fresh canonical OneRobotics A1 MjSpec."""
  return mj.MjSpec.from_file(str(A1_XML))


def _get_entity_spec() -> Any:
  """Load the robot spec without its source keyframe.

  EntityCfg installs the same home pose as ``init_state``. Removing the source
  key from the attached copy avoids carrying two identical keyframes into a
  scene while the packaged MJCF itself remains unchanged.
  """
  spec = get_spec()
  while spec.keys:
    spec.delete(spec.keys[0])
  return spec


# The canonical MJCF carries the validated position actuators. XmlActuatorCfg
# resolves their targets without replacing gains, limits, gears, or dynamics.
A1_ACTUATOR_CFG = XmlActuatorCfg(
  target_names_expr=(r"joint[1-7]-a1_r",),
  command_field="position",
)

A1_HOME_JOINT_POS: dict[str, float] = {
  "joint1-a1_r": 0.0,
  "joint2-a1_r": -0.6,
  "joint3-a1_r": 0.0,
  "joint4-a1_r": 1.0,
  "joint5-a1_r": 0.0,
  "joint6-a1_r": 0.5,
  "joint7-a1_r": 0.0,
}

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  joint_pos=A1_HOME_JOINT_POS,
  joint_vel={".*": 0.0},
)

# Preserve the source model's collision policy. Continuous contact parameters
# remain inherited from the MJCF.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r"Link[1-7]_collision",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=0,
)

A1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(A1_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=1.0,
)


def get_a1_robot_cfg() -> EntityCfg:
  """Return a fresh OneRobotics A1 entity configuration."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=_get_entity_spec,
    articulation=A1_ARTICULATION,
  )


# Matches the validated reach-task joint-position action scale.
A1_ACTION_SCALE: dict[str, float] = {r"joint[1-7]-a1_r": 0.5}
