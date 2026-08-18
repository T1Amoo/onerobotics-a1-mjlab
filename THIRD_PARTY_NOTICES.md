# Third-Party Notices

This file records third-party material and project lineage. It does not change
the license of any file.

## mjlab and runtime dependencies

This independent project uses
[`mujocolab/mjlab`](https://github.com/mujocolab/mjlab) as a normal package
dependency. mjlab is licensed under Apache-2.0. MuJoCo, MuJoCo Warp, RSL-RL,
PyTorch, uv, and their dependency trees are external dependencies and are not
vendored here; each remains under its own supplied license.

## Reach-task lineage

The task behavior was translated from the existing OneRobotics A1 Isaac Lab
reach integration in [`katazen/onerobot_h1`](https://github.com/katazen/onerobot_h1)
to current mjlab manager-based APIs. That source implementation contains files
derived from Isaac Lab reach templates with retained Isaac Lab Project
Developers BSD-3-Clause notices. The complete Isaac Lab BSD-3-Clause text is
included at `LICENSES/IsaacLab-BSD-3-Clause.txt` for attribution continuity.

No Isaac Lab runtime package, scene, table, grid, marker, UI asset, or NVIDIA
binary is bundled in this repository. The new `media/a1.png` render uses only
the packaged OneRobotics A1 model and neutral MuJoCo renderer primitives.

## Robot assets

No third-party mesh, texture, material, or model is bundled with the canonical
A1 MJCF beyond the OneRobotics-owned material covered by `ASSET_LICENSE.md`.
