"""Render the packaged canonical A1 model for the public README."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np
from PIL import Image

from onerobotics_a1_mjlab.a1 import get_spec

mj = cast(Any, mujoco)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=Path("media/a1.png"))
  args = parser.parse_args()

  spec = get_spec()
  spec.visual.global_.offwidth = 960
  spec.visual.global_.offheight = 720
  model = spec.compile()
  data = mj.MjData(model)
  mj.mj_resetDataKeyframe(model, data, model.key("home").id)
  mj.mj_forward(model, data)

  renderer = mj.Renderer(model, height=720, width=960)
  camera = mj.MjvCamera()
  camera.type = mj.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = (0.0, 0.0, 0.25)
  camera.distance = 0.9
  camera.azimuth = 135.0
  camera.elevation = -18.0
  renderer.update_scene(data, camera=camera)
  image = renderer.render()
  renderer.close()

  # MuJoCo's empty background is black. Composite only those background pixels
  # onto a neutral gradient; the rendered A1 pixels remain untouched.
  height, width, _ = image.shape
  vertical = np.linspace(0.0, 1.0, height)[:, None, None]
  upper = np.array([248.0, 249.0, 250.0])[None, None, :]
  lower = np.array([218.0, 224.0, 229.0])[None, None, :]
  background = np.broadcast_to(
    upper * (1.0 - vertical) + lower * vertical,
    (height, width, 3),
  )
  luminance = image.max(axis=2, keepdims=True).astype(np.float64)
  foreground_alpha = np.clip(luminance / 10.0, 0.0, 1.0)
  composited = (
    image * foreground_alpha + background * (1.0 - foreground_alpha)
  ).astype(np.uint8)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(composited).save(args.output, optimize=True)
  print(f"Wrote {args.output} from the canonical packaged A1 MJCF")


if __name__ == "__main__":
  main()
