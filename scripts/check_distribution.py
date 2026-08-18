"""Verify that wheel and sdist archives contain all release-critical files."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

MESH_FILES = {"base_link.STL", *{f"Link_R{i}.STL" for i in range(1, 8)}}


def _archive_names(path: Path) -> set[str]:
  if path.suffix == ".whl":
    with zipfile.ZipFile(path) as archive:
      return set(archive.namelist())
  if path.name.endswith(".tar.gz"):
    with tarfile.open(path, "r:gz") as archive:
      return {member.name for member in archive.getmembers() if member.isfile()}
  raise ValueError(f"Unsupported distribution archive: {path}")


def _matching_suffixes(names: set[str], suffix: str) -> list[str]:
  return sorted(name for name in names if name.endswith(suffix))


def check_archive(path: Path) -> None:
  """Raise an assertion error if a release artifact is incomplete."""
  names = _archive_names(path)
  xml_suffix = "onerobotics_a1_mjlab/a1/xmls/a1.xml"
  assert len(_matching_suffixes(names, xml_suffix)) == 1

  packaged_meshes = {
    Path(name).name
    for name in names
    if "/onerobotics_a1_mjlab/a1/xmls/assets/" in f"/{name}" and name.endswith(".STL")
  }
  assert packaged_meshes == MESH_FILES

  for required in (
    "LICENSE",
    "LICENSES/CC-BY-4.0.txt",
    "LICENSES/IsaacLab-BSD-3-Clause.txt",
    "ASSET_LICENSE.md",
    "THIRD_PARTY_NOTICES.md",
  ):
    assert _matching_suffixes(names, required), f"{required} missing from {path}"

  forbidden = (".env", "/.venv/", "/wandb/", "/logs/", "__pycache__")
  assert not [name for name in names if any(item in name for item in forbidden)]
  print(f"PASS {path.name}: {len(names)} files, 1 MJCF, 8 STL, legal files present")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("archives", nargs="+", type=Path)
  args = parser.parse_args()
  for archive in args.archives:
    check_archive(archive)


if __name__ == "__main__":
  main()
