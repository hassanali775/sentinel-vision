"""Entity state contracts for persistent identity management (PR-006).

This module defines the 5-state lifecycle enum (``EntityState``) and the frozen
``EntityObservation`` contract representing an entity's state at a single frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sentinel_vision.data.contracts import BoundingBox


class EntityState(Enum):
    """Lifecycle states for a persistent entity tracked over time."""

    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    PREDICTED = "PREDICTED"
    LOST = "LOST"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class EntityObservation:
    """Observation of a persistent entity's state at a single frame.

    Invariants validated in ``__post_init__``:
    - ``entity_id`` and ``frame_id`` must be non-negative.
    - ``bounding_box`` must be ``None`` if and only if ``state == EntityState.LOST``.
      Non-LOST states must carry a valid ``BoundingBox``, and ``LOST`` state
      must not carry a box.
    """

    entity_id: int
    state: EntityState
    bounding_box: BoundingBox | None
    class_label: str
    frame_id: int

    def __post_init__(self) -> None:
        if self.entity_id < 0:
            raise ValueError(f"entity_id ({self.entity_id}) must be non-negative")
        if self.frame_id < 0:
            raise ValueError(f"frame_id ({self.frame_id}) must be non-negative")

        if self.state == EntityState.LOST:
            if self.bounding_box is not None:
                raise ValueError("bounding_box must be None when state is LOST")
        else:
            if self.bounding_box is None:
                raise ValueError(
                    f"bounding_box cannot be None when state is {self.state.name}"
                )
