"""Deterministic event reasoning package (PR-009).

The first layer allowed to apply thresholds, accumulate over time, and
produce judgments. ``EventEngine`` converts per-frame spatial workspace
facts (ADR-0008) into sustained ``Event`` objects with hysteresis, driven by
stateless ``BaseEventRule`` predicates (see
docs/adr/0009-deterministic-event-engine.md).
"""

__all__ = [
    "BaseEventRule",
    "Event",
    "EventEngine",
    "EventStatus",
    "EventType",
    "ProximityHazardRule",
    "ZoneIntrusionRule",
]

from sentinel_vision.events.engine import EventEngine
from sentinel_vision.events.event import Event, EventStatus, EventType
from sentinel_vision.events.rules import (
    BaseEventRule,
    ProximityHazardRule,
    ZoneIntrusionRule,
)
