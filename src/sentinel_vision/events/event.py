"""Event contract for the deterministic event engine (PR-009).

This module defines the ``EventType`` and ``EventStatus`` enums and the frozen
``Event`` contract representing a temporal judgment produced by the event
engine: that a rule's condition held, was sustained, and eventually cleared,
over a span of frames. Every field is validated in ``__post_init__`` so a
malformed event is rejected at construction, following the project's contract
validation convention (see docs/adr/0009-deterministic-event-engine.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Kinds of temporal judgment the event engine may produce."""

    PROXIMITY_HAZARD = "PROXIMITY_HAZARD"
    ZONE_INTRUSION = "ZONE_INTRUSION"


class EventStatus(Enum):
    """Lifecycle status of an event: open while its condition holds, then closed."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class Event:
    """A rule's temporal judgment over a span of frames.

    Attributes:
        event_type: The kind of judgment. Determines the shape of
            ``entity_ids`` and whether ``zone_name`` is required.
        entity_ids: The entities involved, ordered. Exactly two
            ``(min_id, max_id)``-convention entity ids for
            ``PROXIMITY_HAZARD``; exactly one entity id for
            ``ZONE_INTRUSION``.
        status: ``OPEN`` while the underlying condition holds, ``CLOSED``
            once it has cleared.
        opened_frame_id: The frame at which the event opened (sustain
            reached). Non-negative.
        closed_frame_id: The frame at which the event closed (clear reached).
            ``None`` if and only if ``status == OPEN``, and strictly greater
            than ``opened_frame_id`` when present.
        zone_name: The named zone for ``ZONE_INTRUSION`` events, ``None``
            for all other types.

    Invariants validated in ``__post_init__``:
    - ``entity_ids`` length must match ``event_type`` (2 for
      ``PROXIMITY_HAZARD``, 1 for ``ZONE_INTRUSION``) and every entity id
      must be non-negative.
    - ``zone_name`` must be a non-empty string if and only if
      ``event_type == ZONE_INTRUSION``.
    - ``opened_frame_id`` must be non-negative.
    - ``closed_frame_id`` must be ``None`` iff ``status == OPEN``, and must
      be strictly greater than ``opened_frame_id`` when ``status == CLOSED``.
    """

    event_type: EventType
    entity_ids: tuple[int, ...]
    status: EventStatus
    opened_frame_id: int
    closed_frame_id: int | None
    zone_name: str | None

    def __post_init__(self) -> None:
        if self.opened_frame_id < 0:
            raise ValueError(
                f"opened_frame_id ({self.opened_frame_id}) must be non-negative"
            )
        for entity_id in self.entity_ids:
            if entity_id < 0:
                raise ValueError(
                    f"entity_id ({entity_id}) must be non-negative"
                )

        if self.event_type is EventType.PROXIMITY_HAZARD:
            if len(self.entity_ids) != 2:
                raise ValueError(
                    "PROXIMITY_HAZARD events must involve exactly two entities, "
                    f"got {len(self.entity_ids)}"
                )
            if self.zone_name is not None:
                raise ValueError(
                    "PROXIMITY_HAZARD events must not carry a zone_name"
                )
        else:  # ZONE_INTRUSION
            if len(self.entity_ids) != 1:
                raise ValueError(
                    "ZONE_INTRUSION events must involve exactly one entity, "
                    f"got {len(self.entity_ids)}"
                )
            if self.zone_name is None or not self.zone_name.strip():
                raise ValueError(
                    "ZONE_INTRUSION events must carry a non-empty zone_name"
                )

        if self.status is EventStatus.OPEN:
            if self.closed_frame_id is not None:
                raise ValueError(
                    "closed_frame_id must be None when status is OPEN"
                )
        else:
            if self.closed_frame_id is None:
                raise ValueError(
                    "closed_frame_id cannot be None when status is CLOSED"
                )
            if self.closed_frame_id <= self.opened_frame_id:
                raise ValueError(
                    "closed_frame_id must be > opened_frame_id: "
                    f"closed_frame_id ({self.closed_frame_id}) <= "
                    f"opened_frame_id ({self.opened_frame_id})"
                )
