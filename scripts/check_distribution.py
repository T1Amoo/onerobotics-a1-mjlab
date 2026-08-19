"""Verify that wheel and sdist archives contain all release-critical files."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

MESH_FILES = {"base_link.STL", *{f"Link_R{i}.STL" for i in range(1, 8)}}
EXPECTED_LICENSE_EXPRESSION = "Apache-2.0 AND CC-BY-4.0"


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


def _read_member(path: Path, suffix: str) -> str:
  names = _archive_names(path)
  matches = _matching_suffixes(names, suffix)
  assert len(matches) == 1, f"Expected one {suffix} in {path}, found {matches}"
  if path.suffix == ".whl":
    with zipfile.ZipFile(path) as archive:
      return archive.read(matches[0]).decode()
  with tarfile.open(path, "r:gz") as archive:
    member = archive.extractfile(matches[0])
    assert member is not None
    return member.read().decode()


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

  metadata_suffix = ".dist-info/METADATA" if path.suffix == ".whl" else "/PKG-INFO"
  metadata = Parser().parsestr(_read_member(path, metadata_suffix))
  assert metadata["License-Expression"] == EXPECTED_LICENSE_EXPRESSION

  forbidden = (
    ".env",
    "/.venv/",
    "/.pytest_cache/",
    "/.ruff_cache/",
    "/wandb/",
    "/logs/",
    "/checkpoints/",
    "MUJOCO_LOG.TXT",
    "__pycache__",
  )
  assert not [name for name in names if any(item in name for item in forbidden)]
  if path.suffix == ".whl":
    assert not [name for name in names if "/tests/" in f"/{name}"]
  print(
    f"PASS {path.name}: {len(names)} files, 1 MJCF, 8 STL, legal files and "
    f"{EXPECTED_LICENSE_EXPRESSION} metadata present"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("archives", nargs="+", type=Path)
  args = parser.parse_args()
  for archive in args.archives:
    check_archive(archive)


if __name__ == "__main__":
  main()
