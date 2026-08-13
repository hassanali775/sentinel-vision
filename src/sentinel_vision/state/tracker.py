"""Persistent entity tracker for stateful entity lifecycle management (PR-006).

This module implements ``PersistentEntityTracker``, wrapping raw frame-level
track outputs (``TrackedDetection`` from PR-005) into persistent entity
observations with a 5-state lifecycle (VISIBLE -> OCCLUDED -> PREDICTED ->
LOST -> RETIRED).
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel_vision.data.contracts import BoundingBox, TrackedDetection
from sentinel_vision.state.entity import EntityObservation, EntityState


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

    Validation:
    Budgets must satisfy ``0 <= occlusion_budget <= prediction_budget <= retirement_budget``.
    An inverted or negative ordering raises a ``ValueError``.
    """

    def __init__(
        self,
        occlusion_budget: int = 1,
        prediction_budget: int = 3,
        retirement_budget: int = 5,
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
                        dx_min = (last.x_min - prev.x_min) / frame_delta
                        dy_min = (last.y_min - prev.y_min) / frame_delta
                        dx_max = (last.x_max - prev.x_max) / frame_delta
                        dy_max = (last.y_max - prev.y_max) / frame_delta
                        box = BoundingBox(
                            x_min=last.x_min + steps * dx_min,
                            y_min=last.y_min + steps * dy_min,
                            x_max=last.x_max + steps * dx_max,
                            y_max=last.y_max + steps * dy_max,
                        )
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
                del self._entities[entity_id]

        return sorted(observations, key=lambda obs: obs.entity_id)
