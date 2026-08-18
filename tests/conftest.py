"""Shared test fixtures."""

from collections.abc import Iterator

import pytest
from mjlab.entity import Entity

from onerobotics_a1_mjlab.a1 import get_a1_robot_cfg


@pytest.fixture(scope="module")
def a1_entity() -> Iterator[Entity]:
  """Construct the packaged A1 entity once per test module."""
  entity = Entity(get_a1_robot_cfg())
  yield entity


@pytest.fixture(scope="module")
def a1_model(a1_entity: Entity):
  """Compile the packaged A1 entity model."""
  return a1_entity.compile()
