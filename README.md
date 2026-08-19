# OneRobotics A1 for mjlab

[![CI](https://github.com/T1Amoo/onerobotics-a1-mjlab/actions/workflows/ci.yml/badge.svg)](https://github.com/T1Amoo/onerobotics-a1-mjlab/actions/workflows/ci.yml)

![OneRobotics A1 in MuJoCo](media/a1.png)

This is the **official OneRobotics A1 integration for mjlab**. It is an
independent OneRobotics project that uses
[`mujocolab/mjlab`](https://github.com/mujocolab/mjlab) as a package dependency,
following the external-project approach recommended by the mjlab maintainers.
It does not add the A1 to mjlab core.

OneRobotics develops the **OneRobotics A1**, a 7-DoF fixed-base robotic arm.
Company information is available at <http://www.onerobot.com/>.

## Overview

The project packages the canonical position-controlled right-arm MuJoCo MJCF,
its eight STL meshes, an mjlab `EntityCfg`, and an end-effector pose-reaching
task. The task samples joint-limit-valid configurations and uses native MuJoCo
forward kinematics to produce targets that are kinematically reachable by
construction in both position and orientation.

Version 0.1.0 is a public alpha for the canonical right arm.

## Features

- Canonical OneRobotics A1 7-DoF right-arm MJCF and meshes
- Existing seven XML position actuators with validated gains and limits
- Portable mjlab `EntityCfg` independent of a source checkout
- Registered `Mjlab-Reach-OneRobotics-A1` task
- 7-dimensional joint-position action space
- 0.1 rad/step position-target rate limit for robust CPU/GPU execution
- Closed-loop position and orientation observations
- Multiplicative multi-scale pose reward
- RSL-RL PPO baseline and standard mjlab `train`/`play` commands
- GitHub-hosted CPU checks and separately recorded local NVIDIA GPU validation
- Explicit Apache-2.0 code and CC BY 4.0 model-asset licensing

## Quick Start

```bash
git clone https://github.com/T1Amoo/onerobotics-a1-mjlab.git
cd onerobotics-a1-mjlab
uv sync --frozen
uv run play Mjlab-Reach-OneRobotics-A1 --agent zero
```

Use `--agent random` to exercise random joint actions:

```bash
uv run play Mjlab-Reach-OneRobotics-A1 --agent random
```

## Installation

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) first.
The project supports the same Python range as mjlab 1.6: Python 3.10 through
3.13. `uv` installs `mjlab>=1.6.0,<2.0.0` from the normal package index; there
is no sibling checkout, file URL, or editable mjlab dependency.

```bash
uv sync --frozen
uv run list-envs | grep Mjlab-Reach-OneRobotics-A1
```

## Available Tasks

| Task ID | Robot | Description |
| --- | --- | --- |
| `Mjlab-Reach-OneRobotics-A1` | OneRobotics A1 | Reach a kinematically reachable end-effector pose |

The single task registration provides separate training and play configs via
mjlab's `env_cfg`, `play_env_cfg`, and `rl_cfg` registry mechanism; no separate
`-Play` task ID is required.

## How it works

```text
a1.xml + 8 STL meshes
          |
          v
get_a1_robot_cfg() + XmlActuatorCfg
          |
          v
     mjlab EntityCfg
          |
          v
    Reach environment
      |           |
      v           v
ReachablePose   observations / actions / rewards
    Command          |
      \              /
       v            v
          RSL-RL PPO
              |
              v
     mjlab task registry
              |
              v
        train / play CLI
```

| Path | Responsibility |
| --- | --- |
| `a1/xmls/` | Canonical MJCF and eight runtime STL meshes |
| `a1/a1_constants.py` | Reusable A1 `EntityCfg`, XML actuators, collisions, and home state |
| `reach/env_cfg.py` | Reach scene, observations, actions, commands, rewards, and episode settings |
| `reach/mdp/` | Reachable target sampling and the task-specific MDP terms |
| `reach/rl_cfg.py` | RSL-RL PPO baseline |
| package `__init__.py` | `Mjlab-Reach-OneRobotics-A1` registration through the `mjlab.tasks` entry point |

Robot integration and task logic are intentionally separate. A future A1 task
should reuse `get_a1_robot_cfg()` from `a1/` and place its task-specific terms
in a new sibling package rather than duplicate the MJCF integration.

## Task design and conventions

Targets are kinematically reachable by construction: every four seconds, the
command term samples a joint-limit-valid seven-joint configuration from the
central 80% of each joint's XML range, then runs native MuJoCo forward
kinematics to obtain the matching end-effector position and orientation. This
avoids asking the policy to reach arbitrary poses outside the arm's kinematic
workspace; it does not guarantee collision-free targets or paths.

The public coordinate and control contract is:

- Joint order: `joint1-a1_r` through `joint7-a1_r`.
- End-effector site: `end_effector`, colocated with the Link7 terminal frame.
- Target pose: `[x, y, z, qw, qx, qy, qz]` in the robot base frame; position is
  in meters and quaternions use scalar-first `(w, x, y, z)` order with a
  non-negative scalar component.
- Pose error: `[target_position - current_position,
  target_quaternion * inverse(current_quaternion)]`, also in the base frame and
  with the quaternion standardized to a non-negative scalar component.
- Action: a seven-dimensional joint-position command in joint order. Before
  rate limiting, the target is `home + 0.5 * action`; the target may move by at
  most 0.1 rad per environment step.
- Timing: `timestep=1/60 s`, `decimation=1`, so simulation and control both run
  at 60 Hz. Targets resample every 4 s (240 control steps).
- Episodes: training uses an 8 s timeout, treated as a truncation by mjlab's
  default infinite-horizon semantics; play uses an effectively infinite
  `1e9 s` timeout. There are no failure terminations.

Actor and critic observations are each 35-dimensional: relative joint position
(7), relative joint velocity (7), previous action (7), target pose (7), and
end-effector pose error (7). During training, only actor joint position and
velocity receive uniform noise (respectively +/-0.01 rad and +/-0.05 rad/s).
The critic uses the same semantic terms without noise; it is not a privileged or
asymmetric critic. Play disables actor corruption as well.

| Reward term | Weight | Purpose |
| --- | ---: | --- |
| `position_error` | -0.5 | Penalize absolute position error |
| `orientation_error` | -0.3 | Penalize shortest-angle orientation error |
| `pose_tracking_coarse` | +0.5 | Broad multiplicative position-orientation shaping |
| `pose_tracking_fine` | +0.5 | Precision shaping near the target |
| `action_rate_l2` | -0.01 | Smooth policy commands |
| `joint_vel_l2` | -0.001 | Suppress unnecessary joint motion |
| `joint_pos_limits` | -1.0 | Discourage joint-limit violations |

These are the coefficients in `env_cfg.py`; mjlab applies its default
environment-step-duration scaling to the combined reward.

## Training

Start the public PPO baseline:

```bash
uv run train Mjlab-Reach-OneRobotics-A1
```

For a larger single-GPU run:

```bash
uv run train Mjlab-Reach-OneRobotics-A1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000
```

Runs are written under `logs/rsl_rl/onerobotics_a1_reach/` using TensorBoard.
The v0.1.0 configuration is a reasonable PPO baseline; this project does not
claim policy convergence without a separately reported training evaluation.

```bash
uv run tensorboard --logdir logs/rsl_rl
```

## Evaluation

Use zero or random actions without a checkpoint:

```bash
uv run play Mjlab-Reach-OneRobotics-A1 --agent zero
uv run play Mjlab-Reach-OneRobotics-A1 --agent random
```

To replay a locally trained policy, pass the checkpoint accepted by the current
mjlab CLI, for example:

```bash
uv run play Mjlab-Reach-OneRobotics-A1 \
  --checkpoint-file logs/rsl_rl/onerobotics_a1_reach/RUN_DIRECTORY/model_N.pt
```

## Robot Model

The only bundled robot is the canonical OneRobotics A1 single right arm:

- fixed base;
- seven revolute joints;
- seven MJCF position actuators;
- eight STL meshes;
- Link7 terminal-frame `end_effector` site;
- home joint configuration `[0, -0.6, 0, 1, 0, 0.5, 0]`.

Dual-arm, left-arm, raw, and redundant motor-control variants are intentionally
not included. The model originates from
<https://github.com/katazen/onerobot_h1>.

The canonical MJCF already contains the validated position actuators. The
integration therefore uses `XmlActuatorCfg` to resolve their position targets;
it does not replace the source gains, force/control limits, gears, armature, or
dynamics. Link1-Link7 collision geoms remain enabled with three-dimensional
contacts, while friction and continuous solver parameters stay inherited from
the MJCF. The fidelity suite verifies the resulting model against the frozen
canonical source snapshot.

`get_spec()` always returns a fresh copy of the packaged canonical MJCF,
including its `home` keyframe. For scene attachment only, `EntityCfg` removes
that keyframe from a second in-memory copy because mjlab installs the equivalent
`init_state` keyframe. This avoids duplicate scene keys and never mutates the
packaged `a1.xml`.

## Validation

See [`VALIDATION.md`](VALIDATION.md) for the tested MuJoCo/mjlab versions,
55-group physical-model comparison, GitHub-hosted CPU checks, separately
recorded local NVIDIA GPU and training smoke results, and isolated wheel/sdist
checks.

## Compatibility and status

| Scope | Support and evidence |
| --- | --- |
| Python | 3.10-3.13; full release CI on 3.13 and focused compatibility smoke on 3.10-3.12 |
| mjlab | `>=1.6.0,<2.0.0`; the complete suite is validated against the minimum 1.6.0 release |
| Robot | Canonical fixed-base OneRobotics A1 right arm only |
| Runtime | CPU release checks in GitHub Actions; NVIDIA GPU simulation/training validated locally |

This alpha release is a simulation and training reference. It does not claim a
converged policy, real-robot deployment, or production readiness.

## Licensing

- Code and mjlab integration: **Apache-2.0** (`LICENSE`)
- OneRobotics A1 model assets (MJCF and meshes): **CC BY 4.0**
  (`ASSET_LICENSE.md`, `LICENSES/CC-BY-4.0.txt`)

OneRobotics A1 robot assets © 2026 OneRobotics. Redistribution of the model
assets must preserve attribution, source, license link, and a description of
changes. The Python package's Apache-2.0 license does not relicense the bundled
CC BY 4.0 model assets.

## Citation and Acknowledgments

Please cite the upstream mjlab project when using its simulation and training
framework. This integration uses MuJoCo, MuJoCo Warp, RSL-RL, and the external
task-registration mechanism demonstrated by
[`mujocolab/anymal_c_velocity`](https://github.com/mujocolab/anymal_c_velocity).

OneRobotics website: <http://www.onerobot.com/>
