"""Per-frame spatial workspace facts (PR-008).

``WorkspaceModel`` evaluates a frame's ``EntityObservation`` list into
descriptive geometric facts: which zones each entity's bounding-box center
falls inside, and the Euclidean distance between every pair of eligible
entities' centers. It emits no temporal judgment, no thresholds, and no
violations — that layer is explicitly PR-009's responsibility (see
docs/adr/0008-spatial-workspace-model.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sentinel_vision.data.contracts import BoundingBox
from sentinel_vision.spatial.zone import Zone
from sentinel_vision.state.entity import EntityObservation, EntityState

_ELIGIBLE_STATES = frozenset(
    {EntityState.VISIBLE, EntityState.OCCLUDED, EntityState.PREDICTED}
)


@dataclass(frozen=True)
class SpatialFrameObservation:
    """Per-frame spatial facts for one frame of entities.

    Attributes:
        frame_id: The frame these facts describe. Must be non-negative.
        zone_memberships: Maps each eligible entity_id to the names of the
            zones containing its bounding-box center, in workspace
            zone-definition order. An eligible entity whose center is in no
            zone maps to an empty list. LOST (and RETIRED) entities are
            absent entirely.
        pairwise_distances: Euclidean distance between the bounding-box
            centers of every unordered pair of eligible entities, keyed by
            ``(min(entity_id_a, entity_id_b), max(entity_id_a, entity_id_b))``
            so lookup is order-independent. Empty when fewer than two
            eligible entities exist.
    """

    frame_id: int
    zone_memberships: dict[int, list[str]]
    pairwise_distances: dict[tuple[int, int], float]

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(f"frame_id ({self.frame_id}) must be non-negative")


class WorkspaceModel:
    """Per-frame spatial workspace model over a fixed set of zones.

    Constructed with a fixed list of ``Zone`` objects; no two zones may
    share a name, since duplicate names would make membership results
    ambiguous. ``evaluate`` turns one frame's entity observations into a
    ``SpatialFrameObservation`` of pure geometric facts.
    """

    def __init__(self, zones: list[Zone]) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for zone in zones:
            if zone.name in seen:
                duplicates.add(zone.name)
            seen.add(zone.name)
        if duplicates:
            raise ValueError(
                f"duplicate zone name(s) in workspace: {sorted(duplicates)}"
            )
        self._zones: tuple[Zone, ...] = tuple(zones)

    def evaluate(
        self, frame_id: int, observations: list[EntityObservation]
    ) -> SpatialFrameObservation:
        """Compute per-frame spatial facts for ``observations`` at ``frame_id``.

        Deliberate eligibility decision: only ``VISIBLE``, ``OCCLUDED``, and
        ``PREDICTED`` entities contribute spatial facts, because only those
        states carry an active bounding box a geometric fact can be computed
        from. ``LOST`` entities have no bounding box (by the ADR-0006
        invariant) and contribute nothing. ``RETIRED`` entities also carry a
        final box but are terminal bookkeeping records, not active spatial
        entities, and are excluded as well. This is a deliberate decision,
        not an oversight.

        Zone membership is decided by the entity's bounding-box center only,
        never by full polygon overlap — a known simplification (see
        docs/adr/0008-spatial-workspace-model.md). Pairwise distance is the
        Euclidean distance between box centers, keyed by
        ``(min, max)`` entity id so the ordering convention is explicit.
        """
        if frame_id < 0:
            raise ValueError(f"frame_id ({frame_id}) must be non-negative")

        eligible: list[tuple[int, BoundingBox]] = [
            (obs.entity_id, obs.bounding_box)
            for obs in observations
            if obs.state in _ELIGIBLE_STATES and obs.bounding_box is not None
        ]

        zone_memberships: dict[int, list[str]] = {}
        for entity_id, box in eligible:
            cx = (box.x_min + box.x_max) / 2.0
            cy = (box.y_min + box.y_max) / 2.0
            zone_memberships[entity_id] = [
                zone.name for zone in self._zones if zone.contains_point(cx, cy)
            ]

        pairwise_distances: dict[tuple[int, int], float] = {}
        for i in range(len(eligible)):
            id_a, box_a = eligible[i]
            cx_a = (box_a.x_min + box_a.x_max) / 2.0
            cy_a = (box_a.y_min + box_a.y_max) / 2.0
            for j in range(i + 1, len(eligible)):
                id_b, box_b = eligible[j]
                cx_b = (box_b.x_min + box_b.x_max) / 2.0
                cy_b = (box_b.y_min + box_b.y_max) / 2.0
                pairwise_distances[(min(id_a, id_b), max(id_a, id_b))] = math.hypot(
                    cx_a - cx_b, cy_a - cy_b
                )

        return SpatialFrameObservation(
            frame_id=frame_id,
            zone_memberships=zone_memberships,
            pairwise_distances=pairwise_distances,
        )
