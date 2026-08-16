"""Persistent entity tracker for stateful entity lifecycle management (PR-006, PR-007).

This module implements ``PersistentEntityTracker``, wrapping raw frame-level
track outputs (``TrackedDetection`` from PR-005) into persistent entity
observations with a 5-state lifecycle (VISIBLE -> OCCLUDED -> PREDICTED ->
LOST -> RETIRED) and optional spatial re-identification (PR-007).
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel_vision.data.contracts import BoundingBox, TrackedDetection
from sentinel_vision.reidentification.spatial import (
    ReidentificationCandidate,
    SpatialReidentifier,
)
from sentinel_vision.state.entity import EntityObservation, EntityState


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

    def update(
        self, frame_id: int, tracked_detections: list[TrackedDetection]
    ) -> list[EntityObservation]:
        """Update entity state for ``frame_id`` given input ``tracked_detections``.

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

        for td in tracked_detections:
            source_track_id = td.track_id
            if source_track_id in track_to_entity:
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
            else:
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
                box = rec.last_observed_box
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
                if (
                    rec.second_to_last_observed_box is not None
                    and rec.second_to_last_observed_frame_id is not None
                ):
                    steps = k - self._occlusion_budget
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
                        box = _bounded_prediction_box(last, velocity, steps)
                    else:
                        box = rec.last_observed_box
                else:
                    # Fallback: only one observed position exists, hold last known box
                    box = rec.last_observed_box

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

