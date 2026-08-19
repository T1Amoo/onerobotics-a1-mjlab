"""Observations for the OneRobotics A1 reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_inv, quat_mul, quat_unique

from .commands import ReachablePoseCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_command(env: ManagerBasedRlEnv, command_name: str) -> ReachablePoseCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReachablePoseCommand):
    raise TypeError(
      f"Command '{command_name}' must be ReachablePoseCommand, got {type(command)}"
    )
  return command


def target_pose_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return base-frame target pose ``[x, y, z, qw, qx, qy, qz]``.

  Position is in meters and the scalar-first quaternion has a non-negative
  scalar component. The output shape is ``(num_envs, 7)``.
  """
  return _get_command(env, command_name).command


def ee_pose_error_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return base-frame pose error with shape ``(num_envs, 7)``.

  The first three values are target-minus-current position in meters. The last
  four are ``q_target * inverse(q_current)`` in scalar-first order, standardized
  to a non-negative scalar component.
  """
  command = _get_command(env, command_name)
  current_pos_b, current_quat_b = command.current_pose_b()
  position_error = command.command[:, :3] - current_pos_b
  orientation_error = quat_unique(
    quat_mul(command.command[:, 3:], quat_inv(current_quat_b))
  )
  return torch.cat((position_error, orientation_error), dim=-1)
