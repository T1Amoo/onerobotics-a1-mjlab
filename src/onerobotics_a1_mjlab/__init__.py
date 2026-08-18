"""Official OneRobotics A1 integration for mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from .a1 import get_a1_robot_cfg
from .reach import a1_reach_env_cfg, a1_reach_ppo_runner_cfg

TASK_ID = "Mjlab-Reach-OneRobotics-A1"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=a1_reach_env_cfg(),
  play_env_cfg=a1_reach_env_cfg(play=True),
  rl_cfg=a1_reach_ppo_runner_cfg(),
)

__all__ = ["TASK_ID", "get_a1_robot_cfg"]
