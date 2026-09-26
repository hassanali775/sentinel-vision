"""Persistent entity tracker for stateful entity lifecycle management (PR-006, PR-007).

This module implements ``PersistentEntityTracker``, wrapping raw frame-level
track outputs (``TrackedDetection`` from PR-005) into persistent entity
observations with a 5-state lifecycle (VISIBLE -> OCCLUDED -> PREDICTED ->
LOST -> RETIRED) and two-tier spatial re-identification: the active
(OCCLUDED/PREDICTED) entity pool is checked first and the fully RETIRED
candidate pool second (ADR-0011, see
docs/adr/0011-active-entity-spatial-matching.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel_vision.data.contracts import BoundingBox, TrackedDetection
from sentinel_vision.reidentification.spatial import (
    ReidentificationCandidate,
    SpatialReidentifier,
    plausibility_score,
)
from sentinel_vision.state.entity import EntityObservation, EntityState

_ACTIVE_MATCHABLE_STATES = (EntityState.OCCLUDED, EntityState.PREDICTED)


def _bounded_prediction_box(
    last: BoundingBox,
    velocity: tuple[float, float, float, float],
    steps: int,
) -> BoundingBox:
    """Linearly extrapolate ``last`` by ``steps`` frames of ``velocity``.

    Per-edge finite-difference velocity can invert a box when the edges
    diverge over a gap (observed on real video with sparse detections). To
    preserve the box invariant, an axis whose projection would invert is
    re-centered on the extrapolated midpoint keeping the last known size.
    """
    dx_min, dy_min, dx_max, dy_max = velocity
    x_min = last.x_min + steps * dx_min
    x_max = last.x_max + steps * dx_max
    y_min = last.y_min + steps * dy_min
    y_max = last.y_max + steps * dy_max
    if x_min >= x_max:
        mid_x = (x_min + x_max) / 2.0
        half_width = (last.x_max - last.x_min) / 2.0
        x_min, x_max = mid_x - half_width, mid_x + half_width
    if y_min >= y_max:
        mid_y = (y_min + y_max) / 2.0
        half_height = (last.y_max - last.y_min) / 2.0
        y_min, y_max = mid_y - half_height, mid_y + half_height
    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


@dataclass
class _EntityRecord:
    entity_id: int
    current_state: EntityState
    last_observed_box: BoundingBox
    second_to_last_observed_box: BoundingBox | None
    last_observed_frame_id: int
    second_to_last_observed_frame_id: int | None
    frames_since_last_match: int
    source_track_id: int
    class_label: str


class PersistentEntityTracker:
    """Wraps raw tracking output with persistent 5-state entity lifecycle management.

    Maintains active entities through a 5-state transition pipeline:
    ``VISIBLE`` -> ``OCCLUDED`` -> ``PREDICTED`` -> ``LOST`` -> ``RETIRED``.

    Budget Parameters:
    - ``occlusion_budget``: consecutive unobserved frames during which the entity
      is considered OCCLUDED (holding the last known bounding box).
    - ``prediction_budget``: cumulative unobserved frames up to which the entity
      is PREDICTED (extrapolating bounding box linearly using finite differences).
      If only one observed position exists, prediction falls back to holding the last
      known box.
    - ``retirement_budget``: cumulative unobserved frames up to which the entity
      is LOST (bounding box is None). Beyond this threshold, the entity transitions
      to RETIRED, emits a final RETIRED observation, and is purged from internal state.

    Re-identification (PR-007):
    - When ``reidentifier`` is provided, retired entities are added to the candidate
      retention pool upon retirement. When an unmatched track arrives, the tracker
      queries the retention pool before minting a brand-new entity ID; if a candidate
      matches spatial trajectory prediction, the track is re-linked to the candidate's
      original entity ID.

    Re-identification (ADR-0011), two tiers, checked in this order:
    - Tier 1 — active pool: before the retired pool is consulted, unmatched tracks are
      scored against every unclaimed entity that is currently ``OCCLUDED`` or
      ``PREDICTED``, using that entity's box for the current frame. This is what
      recovers a track id re-issued in the middle of a scene by an upstream tracker's
      track expiry during a crossing occlusion, before the entity has aged all the
      way to ``RETIRED``. Matching is one-to-one within a frame and uses the
      reidentifier's own thresholds via the shared ``plausibility_score``.
    - Tier 2 — retired pool: the pre-existing ``SpatialReidentifier.match`` path, then
      minting a new entity ID.
    - Entities in ``LOST`` are in neither tier. A ``LOST`` entity has no box and no
      believed position, so it is not a match target; it becomes eligible only after
      it retires and enters the tier-2 pool. This is deliberate, and it is the one
      case where a nearby detection legitimately mints a new entity.
    - Without a ``reidentifier`` there is no threshold configuration to read, so tier 1
      is skipped entirely and all unmatched tracks go straight to tier 2 / minting.

    Validation:
    Budgets must satisfy ``0 <= occlusion_budget <= prediction_budget <= retirement_budget``.
    An inverted or negative ordering raises a ``ValueError``.
    """

    def __init__(
        self,
        occlusion_budget: int = 1,
        prediction_budget: int = 3,
        retirement_budget: int = 5,
        reidentifier: SpatialReidentifier | None = None,
    ) -> None:
        if occlusion_budget < 0 or prediction_budget < 0 or retirement_budget < 0:
            raise ValueError("All budgets must be non-negative")
        if not (occlusion_budget <= prediction_budget <= retirement_budget):
            raise ValueError(
                f"Budgets must satisfy occlusion_budget ({occlusion_budget}) <= "
                f"prediction_budget ({prediction_budget}) <= "
                f"retirement_budget ({retirement_budget})"
            )

        self._occlusion_budget = occlusion_budget
        self._prediction_budget = prediction_budget
        self._retirement_budget = retirement_budget
        self._reidentifier = reidentifier

        self._entities: dict[int, _EntityRecord] = {}
        self._next_entity_id = 0

    def _match_active_entity(
        self, td: TrackedDetection, matched_entity_ids: set[int]
    ) -> int | None:
        """Return the best active entity for orphan detection ``td``, or ``None``.

        The active-candidate pool is derived, not stored: an entity is a
        candidate exactly when it is unclaimed for this frame, currently
        OCCLUDED or PREDICTED, and ``_predicted_box_for`` still yields a box for
        it at ``frames_since_last_match + 1``. Deriving it means a claim takes
        effect immediately — the claiming caller adds the entity to
        ``matched_entity_ids`` — which is what makes the matching one-to-one
        within a frame without any extra bookkeeping or stale pool state.

        Selection mirrors ``SpatialReidentifier.match`` exactly: score every
        plausible candidate with the shared ``plausibility_score`` using the
        reidentifier's own ``max_distance``/``min_iou``, then take the smallest
        (prediction error, ``entity_id``) pair. Because the winner is chosen by a
        total order over those two keys rather than by iteration order, the
        result is independent of the order entities happen to sit in the dict —
        the same guarantee PR-007's disambiguation test pins for the retired
        pool.

        Returns ``None`` — meaning "fall through to the retired pool, then to
        minting" — when:
        - no reidentifier is configured, so there are no plausibility thresholds
          to apply (active matching is skipped entirely, never a partial match);
        - no unclaimed entity is in an active state;
        - every active entity's predicted box is ``None`` (they are LOST or
          beyond, and a LOST entity is never offered as a match target);
        - the detection is plausible against none of them, including by class.
        """
        reidentifier = self._reidentifier
        if reidentifier is None:
            return None

        best: tuple[float, int] | None = None
        for entity_id, rec in self._entities.items():
            if entity_id in matched_entity_ids:
                continue
            if rec.current_state not in _ACTIVE_MATCHABLE_STATES:
                continue
            candidate_box = self._predicted_box_for(
                rec, rec.frames_since_last_match + 1
            )
            if candidate_box is None:
                continue
            score = plausibility_score(
                td.detection.bounding_box,
                td.detection.class_label,
                candidate_box,
                rec.class_label,
                reidentifier.max_distance,
                reidentifier.min_iou,
            )
            if score is None:
                continue
            key = (score, entity_id)
            if best is None or key < best:
                best = key

        return None if best is None else best[1]

    def _predicted_box_for(
        self, rec: _EntityRecord, frames_unmatched: int
    ) -> BoundingBox | None:
        """Return the box ``rec`` would be reported with after ``frames_unmatched`` misses.

        This is the single shared "where would this entity be" computation of the
        tracker. It is called from two places, and both must agree exactly:

        1. The aging loop in ``update``, with ``frames_unmatched = k`` (the miss
           count the entity just accumulated) for the OCCLUDED/PREDICTED states.
        2. The active-entity candidate pool in ``update``, with
           ``frames_unmatched = rec.frames_since_last_match + 1`` — the count
           this entity *would* reach if it goes unmatched again on the current
           frame. Without this, an active entity would be matched against a
           stale box while the aging loop simultaneously publishes a newer one,
           which is the class of duplicated-logic divergence bug ADR-0011 exists
           to prevent.

        The returned box depends on how far the entity has aged:

        - ``frames_unmatched <= occlusion_budget`` -> OCCLUDED: the last
          observed box is held unchanged.
        - ``occlusion_budget < frames_unmatched <= prediction_budget`` ->
          PREDICTED: the box is extrapolated by the frame-delta-normalized
          finite-difference velocity of the two most recent observations, for
          ``frames_unmatched - occlusion_budget`` prediction steps. With fewer
          than two observations, or a non-positive frame gap, the last observed
          box is held instead.
        - Anything beyond ``prediction_budget`` (LOST, or beyond to RETIRED) ->
          ``None``: there is no box, because the entity is no longer believed to
          be at a known position. Note the RETIRED *observation* still reports
          the last known box for bookkeeping, but it is not a position the
          entity is expected at, so it must never be offered as a match target.
        """
        if frames_unmatched <= self._occlusion_budget:
            return rec.last_observed_box

        if frames_unmatched > self._prediction_budget:
            return None

        if (
            rec.second_to_last_observed_box is None
            or rec.second_to_last_observed_frame_id is None
        ):
            return rec.last_observed_box

        steps = frames_unmatched - self._occlusion_budget
        frame_delta = rec.last_observed_frame_id - rec.second_to_last_observed_frame_id
        if frame_delta <= 0:
            return rec.last_observed_box

        last = rec.last_observed_box
        prev = rec.second_to_last_observed_box
        velocity = (
            (last.x_min - prev.x_min) / frame_delta,
            (last.y_min - prev.y_min) / frame_delta,
            (last.x_max - prev.x_max) / frame_delta,
            (last.y_max - prev.y_max) / frame_delta,
        )
        return _bounded_prediction_box(last, velocity, steps)

    def update(
        self, frame_id: int, tracked_detections: list[TrackedDetection]
    ) -> list[EntityObservation]:
        """Update entity state for ``frame_id`` given input ``tracked_detections``.

        The per-frame flow is deliberately ordered, and the order is the whole
        point of ADR-0011 (see docs/adr/0011-active-entity-spatial-matching.md):

        1. Detections whose ``track_id`` is already known to an entity are
           processed as direct VISIBLE matches and claim that entity.
        2. An active-candidate pool is built from every unclaimed entity that is
           currently OCCLUDED or PREDICTED, using the exact box the aging loop
           would publish for it on this very frame.
        3. Each remaining detection (an "orphan": an unseen ``track_id``, in
           input order) is scored against that pool with the reidentifier's own
           thresholds. The closest plausible candidate — ties broken by lowest
           ``entity_id`` — claims the detection, one-to-one.
        4. A claimed detection re-links to the active entity's ``entity_id``,
           taking over that entity's ``source_track_id``, and the entity returns
           to VISIBLE.
        5. An unclaimed detection falls through unchanged: first to
           ``SpatialReidentifier.match`` over the RETIRED candidate pool, then to
           minting a new entity.
        6. Every entity still unclaimed ages exactly as it did before.

        Active-entity matching requires a ``SpatialReidentifier``, because the
        plausibility thresholds are that object's configuration. When the
        tracker is constructed with ``reidentifier=None`` there is no threshold
        configuration to read, so steps 2-4 are skipped entirely for the frame
        and every orphan falls through to the retired-pool/new-entity path. This
        is a deliberate, documented no-op, not a crash and not a silent
        half-applied match.

        Returns one ``EntityObservation`` per active entity in stream order,
        sorted by ``entity_id``.
        """
        if frame_id < 0:
            raise ValueError(f"frame_id ({frame_id}) must be non-negative")

        if self._reidentifier is not None:
            self._reidentifier.purge_expired(frame_id)

        matched_entity_ids: set[int] = set()
        observations: list[EntityObservation] = []

        track_to_entity: dict[int, int] = {
            rec.source_track_id: rec.entity_id for rec in self._entities.values()
        }

        direct_matches: list[TrackedDetection] = []
        orphan_detections: list[TrackedDetection] = []
        for td in tracked_detections:
            if td.track_id in track_to_entity:
                direct_matches.append(td)
            else:
                orphan_detections.append(td)

        for td in direct_matches:
            source_track_id = td.track_id
            entity_id = track_to_entity[source_track_id]
            rec = self._entities[entity_id]
            rec.current_state = EntityState.VISIBLE
            rec.frames_since_last_match = 0
            rec.second_to_last_observed_box = rec.last_observed_box
            rec.last_observed_box = td.detection.bounding_box
            rec.second_to_last_observed_frame_id = rec.last_observed_frame_id
            rec.last_observed_frame_id = frame_id
            rec.class_label = td.detection.class_label
            matched_entity_ids.add(entity_id)
            observations.append(
                EntityObservation(
                    entity_id=entity_id,
                    state=EntityState.VISIBLE,
                    bounding_box=td.detection.bounding_box,
                    class_label=rec.class_label,
                    frame_id=frame_id,
                )
            )

        for td in orphan_detections:
            source_track_id = td.track_id
            active_entity_id = self._match_active_entity(td, matched_entity_ids)

            if active_entity_id is not None:
                rec = self._entities[active_entity_id]
                rec.current_state = EntityState.VISIBLE
                rec.frames_since_last_match = 0
                rec.second_to_last_observed_box = rec.last_observed_box
                rec.last_observed_box = td.detection.bounding_box
                rec.second_to_last_observed_frame_id = rec.last_observed_frame_id
                rec.last_observed_frame_id = frame_id
                rec.source_track_id = source_track_id
                rec.class_label = td.detection.class_label
                matched_entity_ids.add(active_entity_id)
                track_to_entity[source_track_id] = active_entity_id
                observations.append(
                    EntityObservation(
                        entity_id=active_entity_id,
                        state=EntityState.VISIBLE,
                        bounding_box=td.detection.bounding_box,
                        class_label=rec.class_label,
                        frame_id=frame_id,
                    )
                )
                continue

            reid_candidate = (
                self._reidentifier.match(td.detection, frame_id)
                if self._reidentifier is not None
                else None
            )
            if reid_candidate is not None:
                entity_id = reid_candidate.entity_id
                rec = _EntityRecord(
                    entity_id=entity_id,
                    current_state=EntityState.VISIBLE,
                    last_observed_box=td.detection.bounding_box,
                    second_to_last_observed_box=reid_candidate.last_known_box,
                    last_observed_frame_id=frame_id,
                    second_to_last_observed_frame_id=reid_candidate.last_observed_frame_id,
                    frames_since_last_match=0,
                    source_track_id=source_track_id,
                    class_label=td.detection.class_label,
                )
            else:
                entity_id = self._next_entity_id
                self._next_entity_id += 1
                rec = _EntityRecord(
                    entity_id=entity_id,
                    current_state=EntityState.VISIBLE,
                    last_observed_box=td.detection.bounding_box,
                    second_to_last_observed_box=None,
                    last_observed_frame_id=frame_id,
                    second_to_last_observed_frame_id=None,
                    frames_since_last_match=0,
                    source_track_id=source_track_id,
                    class_label=td.detection.class_label,
                )
            self._entities[entity_id] = rec
            matched_entity_ids.add(entity_id)
            track_to_entity[source_track_id] = entity_id
            observations.append(
                EntityObservation(
                    entity_id=entity_id,
                    state=EntityState.VISIBLE,
                    bounding_box=td.detection.bounding_box,
                    class_label=rec.class_label,
                    frame_id=frame_id,
                )
            )

        for entity_id, rec in list(self._entities.items()):
            if entity_id in matched_entity_ids:
                continue

            rec.frames_since_last_match += 1
            k = rec.frames_since_last_match

            if k <= self._occlusion_budget:
                rec.current_state = EntityState.OCCLUDED
                box = self._predicted_box_for(rec, k)
                observations.append(
                    EntityObservation(
                        entity_id=entity_id,
                        state=EntityState.OCCLUDED,
                        bounding_box=box,
                        class_label=rec.class_label,
                        frame_id=frame_id,
                    )
                )
            elif k <= self._prediction_budget:
                rec.current_state = EntityState.PREDICTED
                box = self._predicted_box_for(rec, k)
                observations.append(
                    EntityObservation(
                        entity_id=entity_id,
                        state=EntityState.PREDICTED,
                        bounding_box=box,
                        class_label=rec.class_label,
                        frame_id=frame_id,
                    )
                )
            elif k <= self._retirement_budget:
                rec.current_state = EntityState.LOST
                observations.append(
                    EntityObservation(
                        entity_id=entity_id,
                        state=EntityState.LOST,
                        bounding_box=None,
                        class_label=rec.class_label,
                        frame_id=frame_id,
                    )
                )
            else:
                rec.current_state = EntityState.RETIRED
                box = rec.last_observed_box
                observations.append(
                    EntityObservation(
                        entity_id=entity_id,
                        state=EntityState.RETIRED,
                        bounding_box=box,
                        class_label=rec.class_label,
                        frame_id=frame_id,
                    )
                )
                if self._reidentifier is not None:
                    if (
                        rec.second_to_last_observed_box is not None
                        and rec.second_to_last_observed_frame_id is not None
                    ):
                        frame_delta = (
                            rec.last_observed_frame_id
                            - rec.second_to_last_observed_frame_id
                        )
                        if frame_delta > 0:
                            last = rec.last_observed_box
                            prev = rec.second_to_last_observed_box
                            velocity = (
                                (last.x_min - prev.x_min) / frame_delta,
                                (last.y_min - prev.y_min) / frame_delta,
                                (last.x_max - prev.x_max) / frame_delta,
                                (last.y_max - prev.y_max) / frame_delta,
                            )
                        else:
                            velocity = (0.0, 0.0, 0.0, 0.0)
                    else:
                        velocity = (0.0, 0.0, 0.0, 0.0)

                    candidate = ReidentificationCandidate(
                        entity_id=entity_id,
                        last_known_box=box,
                        velocity=velocity,
                        retired_frame_id=frame_id,
                        last_observed_frame_id=rec.last_observed_frame_id,
                        class_label=rec.class_label,
                    )
                    self._reidentifier.add_candidate(candidate)

                del self._entities[entity_id]

        return sorted(observations, key=lambda obs: obs.entity_id)

