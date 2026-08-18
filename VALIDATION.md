# Validation

This document records the v0.1.0 engineering validation of the standalone
OneRobotics A1 mjlab project. Results were rerun on 2026-08-18 against the
normal PyPI dependency `mjlab==1.6.0`; no sibling mjlab checkout was used by
the project environment or isolated distribution tests.

## Reference snapshots

- Canonical source: `katazen/onerobot_h1` commit
  `ca6d705f37b0dc296bfe7f33f7c83d780c3d3a70`
- Canonical model: OneRobotics A1 right arm, position-control MJCF
- mjlab source inspected: commit
  `0fb8a681136be94ffc636a3dd423cabb97d91f10`
- `mujocolab/anymal_c_velocity` structure inspected: commit
  `248ee73af83ac466a525758469e653217cdc5322`
- Runtime: Python 3.13.9, MuJoCo 3.11.0, mjlab 1.6.0
- GPU validation: NVIDIA GeForce RTX 4060 Laptop GPU

## Canonical model

| Check | Result |
| --- | --- |
| Joint count | 7 |
| Body count | 9 compiled bodies: world, fixed-base wrapper, Link1-Link7 |
| Actuator count | 7 XML position actuators |
| Mesh count | 8 STL files |
| Native `MjSpec.from_file()` | PASS |
| Native MuJoCo compile | PASS |
| Native MuJoCo 100-step simulation | PASS |
| Mesh path resolution | PASS |

## Physics fidelity

The live source-to-candidate comparison passed all 55 named groups:

- 7 joint groups: type, position, axis, range, stiffness, `solref`, `solimp`;
- 3 DoF groups: armature, damping, friction loss;
- 6 body groups: pose, quaternion, mass, inertia, inertial position, inertial
  quaternion;
- 12 geom groups: type, contact flags, contact dimension, priority, friction,
  solver parameters, size, pose, quaternion, and RGBA;
- 12 actuator groups: transmission/dynamics/gain/bias types, transmission IDs,
  gear, gain/bias parameters, control/force ranges, and limit flags;
- 2 home-keyframe groups;
- 3 name groups;
- timestep and integrator;
- 8 source-to-package STL SHA256 checks.

```text
Physics/model comparison: 55/55 PASS
Physics/model parameter differences from canonical source: NONE
```

The checked-in `tests/data/a1_fidelity_reference.json` lets CI repeat all 55
checks without downloading another repository. A live comparison against an
authoritative source checkout was also run with:

```bash
A1_CANONICAL_SOURCE=<onerobot_h1>/source/h1_reach/h1_reach/assets/mjcf/A1/a1_right_position.xml \
  uv run pytest -q tests/test_a1_model_fidelity.py
```

Permitted structural adaptations are limited to package-relative mesh paths,
descriptive names, an mjlab fixed-base wrapper, removal of non-robot demo
elements, and a non-physical Link7 `end_effector` site. The seven actuator
definitions and every compared model parameter remain unchanged.

mjlab warns while attaching the entity that child-level `<option>` values do
not propagate. The task therefore explicitly supplies the same validated
`timestep=1/60` and `integrator=implicitfast` through `MujocoCfg`; the compiled
runtime model was checked to carry those values. The warning does not indicate
a runtime parameter difference.

## Reach task

The registered task is `Mjlab-Reach-OneRobotics-A1`.

- Action dimension: 7 joint-position commands.
- Action scale: 0.5 around the home configuration.
- Position-target rate limit: 0.1 rad per environment step. This task-layer
  limit prevents discontinuous policy outputs from destabilizing MuJoCo Warp;
  it does not modify the MJCF, actuator gains, force limits, or other physics.
- Observation dimensions: 35 for actor and 35 for critic.
- Targets: sampled inside 80% of the joint ranges and transformed with native
  MuJoCo forward kinematics, so position and orientation are reachable by
  construction.
- Reward: linear position/orientation penalties plus multiplicative coarse and
  fine pose kernels.
- Termination: fixed-duration timeout only.

The entry point was discovered from installed metadata, `list-envs` listed the
task, and both real CLI launch paths entered their Viser simulation loops:

```bash
uv run play Mjlab-Reach-OneRobotics-A1 --agent zero
uv run play Mjlab-Reach-OneRobotics-A1 --agent random
```

The interactive viewers were deliberately stopped after their bounded launch
checks; the automated environment tests independently execute zero and random
actions and assert finite observations, rewards, and states.

## CPU and GPU runtime

| Check | Result |
| --- | --- |
| CPU entity construction/reset/steps | PASS |
| CPU reach environment construction | PASS |
| CPU zero/random action smoke | PASS |
| GPU reach environment construction | PASS |
| GPU 256-env, 500-step smoke | PASS |
| GPU zero actions, 100 steps | PASS |
| GPU random actions, 400 steps | PASS |
| Finite GPU observations/rewards/state | PASS |

## Training smoke

The public CLI completed three PPO updates on GPU with 256 environments:

```bash
uv run train Mjlab-Reach-OneRobotics-A1 \
  --env.scene.num-envs 256 \
  --env.seed 42 \
  --agent.max-iterations 3 \
  --agent.save-interval 1 \
  --enable-nan-guard True
```

The runner collected 18,432 environment steps, completed rollout and
backpropagation, reported finite reward/value/surrogate statistics, wrote a
TensorBoard event file, and produced `model_0.pt`, `model_1.pt`, and
`model_2.pt`. No NaN dump was produced. This is a functional training smoke,
not a convergence claim.

## Tests and tooling

```text
Ruff format/check: PASS
Pyright: 0 errors, 0 warnings, 0 informations
pytest: 68 passed
```

The pytest total includes 55 independently parameterized fidelity assertions,
native model tests, Entity tests, task registration/configuration, reachable
target checks, CPU simulation, and zero/random action smoke.

## Package and isolated installation

```text
Wheel build: PASS
sdist build: PASS
Wheel contents: 1 MJCF, 8 STL, required legal files — PASS
sdist contents: 1 MJCF, 8 STL, required legal files — PASS
Isolated wheel installation: PASS
Isolated sdist installation: PASS
```

Both distributions were installed from an arbitrary temporary working
directory. In each environment, `mjlab.__file__` resolved under the isolated
uv package cache's `site-packages/mjlab/__init__.py`, not to the workspace or
the sibling source checkout. The installed package then loaded and compiled
the A1 MJCF, constructed `EntityCfg`/`Entity`, ran native MuJoCo steps, and
discovered the registered reach task.

## CI scope

GitHub Actions runs the feasible CPU release checks on a hosted runner: frozen
uv installation, Ruff, Pyright, pytest, wheel/sdist builds, archive inspection,
and an isolated wheel-install smoke. GPU simulation and training remain local
release validation because hosted CPU runners do not provide NVIDIA GPUs.
