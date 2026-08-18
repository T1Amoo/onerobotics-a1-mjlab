"""Rewards for the OneRobotics A1 reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from .observations import _get_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def position_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Euclidean end-effector position error."""
  command = _get_command(env, command_name)
  current_pos_b, _ = command.current_pose_b()
  return torch.linalg.vector_norm(command.command[:, :3] - current_pos_b, dim=-1)


def orientation_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Shortest end-effector orientation error in radians."""
  command = _get_command(env, command_name)
  _, current_quat_b = command.current_pose_b()
  return quat_error_magnitude(command.command[:, 3:], current_quat_b)


def pose_tracking_tanh(
  env: ManagerBasedRlEnv,
  command_name: str,
  pos_std: float,
  ori_std: float,
) -> torch.Tensor:
  """Multiplicative position-orientation tracking kernel.

  A high value requires both errors to be small; one objective cannot compensate
  for an arbitrarily poor value of the other.
  """
  pos_score = 1.0 - torch.tanh(position_error(env, command_name) / pos_std)
  ori_score = 1.0 - torch.tanh(orientation_error(env, command_name) / ori_std)
  return pos_score * ori_score
