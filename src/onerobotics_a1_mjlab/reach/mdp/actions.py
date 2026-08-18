"""A1 reach-task action terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RateLimitedJointPositionAction(JointPositionAction):
  """Joint-position action with a per-step target slew-rate limit.

  The canonical XML position actuators remain responsible for control. This
  term only prevents discontinuous policy outputs from instantaneously moving
  their position targets by multiple radians in one 1/60 s simulation step.
  """

  cfg: RateLimitedJointPositionActionCfg

  def __init__(
    self,
    cfg: RateLimitedJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)
    if isinstance(self.offset, torch.Tensor):
      self._initial_target = self.offset.clone()
    else:
      self._initial_target = torch.full_like(
        self._processed_actions,
        float(self.offset),
      )
    self._limited_target = self._initial_target.clone()

  @property
  def limited_target(self) -> torch.Tensor:
    """Position target after scaling, offset, clipping, and rate limiting."""
    return self._limited_target

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    target_delta = torch.clamp(
      self._processed_actions - self._limited_target,
      min=-self.cfg.max_target_delta,
      max=self.cfg.max_target_delta,
    )
    self._limited_target.add_(target_delta)
    self._processed_actions = self._limited_target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._limited_target[env_ids] = self._initial_target[env_ids]


@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
  """Configuration for rate-limited A1 position targets."""

  max_target_delta: float = 0.1
  """Maximum position-target change per environment step, in radians."""

  def build(self, env: ManagerBasedRlEnv) -> RateLimitedJointPositionAction:
    return RateLimitedJointPositionAction(self, env)
