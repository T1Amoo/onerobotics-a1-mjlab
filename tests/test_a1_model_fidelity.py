"""55-group fidelity regression against the canonical OneRobotics A1 source."""

import json
import os
from pathlib import Path

import pytest
from scripts.model_fidelity import compare_groups, extract_groups, values_match

from onerobotics_a1_mjlab.a1 import A1_XML

REFERENCE_PATH = Path(__file__).parent / "data" / "a1_fidelity_reference.json"


def _reference_groups() -> dict[str, object]:
  payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
  groups = payload["groups"]
  assert isinstance(groups, dict)
  assert len(groups) == 55
  return groups


@pytest.mark.parametrize("group_name", sorted(_reference_groups()))
def test_candidate_matches_canonical_reference(group_name: str) -> None:
  reference = _reference_groups()
  candidate = extract_groups(A1_XML, canonical_source=False)
  assert values_match(candidate[group_name], reference[group_name]), group_name


def test_optional_live_source_comparison() -> None:
  source_env = os.environ.get("A1_CANONICAL_SOURCE")
  if source_env is None:
    pytest.skip("Set A1_CANONICAL_SOURCE for a live source-to-candidate comparison")

  source = extract_groups(Path(source_env), canonical_source=True)
  candidate = extract_groups(A1_XML, canonical_source=False)
  baseline = _reference_groups()
  assert compare_groups(source, baseline) == []
  assert compare_groups(candidate, source) == []
