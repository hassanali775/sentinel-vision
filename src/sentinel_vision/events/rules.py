"""Stateless rule predicates for the deterministic event engine (PR-009).

``BaseEventRule`` is the contract every rule fulfills: a single per-frame
predicate, ``condition_holds``, which maps a ``SpatialFrameObservation`` to
the set of keys for which the rule's condition currently holds. Each key maps
to one independently-tracked event — a hazard key is one ``(min, max)`` entity
pair, an intrusion key is one entity id. Rules are stateless and hold no
frame-to-frame memory; all temporal accumulation lives in ``EventEngine``
(see docs/adr/0009-deterministic-event-engine.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sentinel_vision.events.event import EventType
from sentinel_vision.spatial.workspace import SpatialFrameObservation

_KeyT = TypeVar("_KeyT")


class BaseEventRule(ABC, Generic[_KeyT]):
    """A stateless per-frame condition over spatial workspace facts.

    Subclasses implement exactly one abstract method, ``condition_holds``,
    which returns the set of keys whose condition holds in ``spatial``.
    ``event_type`` and ``zone_name`` describe the events those keys produce
    so the engine can build them without knowing rule specifics.
    """

    event_type: EventType
    zone_name: str | None = None

    @abstractmethod
    def condition_holds(self, spatial: SpatialFrameObservation) -> set[_KeyT]:
        """Return the keys whose condition holds in ``spatial``.

        ``spatial`` is a single frame's descriptive spatial facts
        (``zone_memberships`` and ``pairwise_distances``), never anything the
        rule itself accumulates over time. The returned set is the rule's
        current truth value per key for exactly this frame.
        """


class ProximityHazardRule(BaseEventRule[tuple[int, int]]):
    """Holds for every entity pair closer than ``threshold_px``.

    A pair's key is its order-normalized ``(min_id, max_id)`` tuple — the
    exact key convention of ``SpatialFrameObservation.pairwise_distances`` —
    so each unordered pair maps to exactly one event. The comparison is
    strictly ``distance < threshold_px``: a pair exactly at the threshold is
    not a hazard, which keeps the open/close boundary deterministic (see
    docs/adr/0008-spatial-workspace-model.md on pixel-space caveats).
    """

    event_type = EventType.PROXIMITY_HAZARD

    def __init__(self, threshold_px: float) -> None:
        if threshold_px <= 0.0:
            raise ValueError(f"threshold_px ({threshold_px}) must be > 0.0")
        self.threshold_px = threshold_px

    def condition_holds(
        self, spatial: SpatialFrameObservation
    ) -> set[tuple[int, int]]:
        return {
            pair
            for pair, distance in spatial.pairwise_distances.items()
            if distance < self.threshold_px
        }


class ZoneIntrusionRule(BaseEventRule[int]):
    """Holds for every entity whose bounding-box center is inside ``zone_name``.

    A key is the entity id itself. Membership is decided by the entity's box
    center against the named zone using the same center-point rule as
    ``WorkspaceModel`` (see docs/adr/0008-spatial-workspace-model.md).
    """

    event_type = EventType.ZONE_INTRUSION

    def __init__(self, zone_name: str) -> None:
        if not zone_name.strip():
            raise ValueError("zone_name must be a non-empty string")
        self.zone_name = zone_name

    def condition_holds(self, spatial: SpatialFrameObservation) -> set[int]:
        return {
            entity_id
            for entity_id, zone_names in spatial.zone_memberships.items()
            if self.zone_name in zone_names
        }
