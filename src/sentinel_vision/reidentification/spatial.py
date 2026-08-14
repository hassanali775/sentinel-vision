"""Spatial/motion-only re-identification and candidate retention pool (PR-007).

This module implements ``ReidentificationCandidate`` and ``SpatialReidentifier``
to re-link newly detected tracks to recently retired entities based on trajectory
extrapolation (finite-difference velocity) without appearance features or heavy
dependencies (see docs/adr/0007-spatial-reidentification.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sentinel_vision.data.contracts import BoundingBox, Detection
from sentinel_vision.evaluation.geometry import iou


@dataclass(frozen=True)
class ReidentificationCandidate:
    """A retired entity retained for potential spatial re-identification.

    Holds the retired entity's ID, last known observed bounding box, per-frame velocity,
    retired frame ID, last observed frame ID, and class label.

    Invariants validated in ``__post_init__``:
    - ``entity_id``, ``retired_frame_id``, and ``last_observed_frame_id`` must be non-negative.
    - ``last_observed_frame_id`` must be <= ``retired_frame_id``.
    """

    entity_id: int
    last_known_box: BoundingBox
    velocity: tuple[float, float, float, float]
    retired_frame_id: int
    last_observed_frame_id: int
    class_label: str

    def __post_init__(self) -> None:
        if self.entity_id < 0:
            raise ValueError(f"entity_id ({self.entity_id}) must be non-negative")
        if self.retired_frame_id < 0:
            raise ValueError(
                f"retired_frame_id ({self.retired_frame_id}) must be non-negative"
            )
        if self.last_observed_frame_id < 0:
            raise ValueError(
                f"last_observed_frame_id ({self.last_observed_frame_id}) must be non-negative"
            )
        if self.last_observed_frame_id > self.retired_frame_id:
            raise ValueError(
                f"last_observed_frame_id ({self.last_observed_frame_id}) cannot exceed "
                f"retired_frame_id ({self.retired_frame_id})"
            )

    def predict_box(self, frame_id: int) -> BoundingBox:
        """Linearly extrapolate bounding box to ``frame_id`` using velocity."""
        if frame_id < self.retired_frame_id:
            raise ValueError(
                f"frame_id ({frame_id}) cannot be before retired_frame_id ({self.retired_frame_id})"
            )
        elapsed = frame_id - self.last_observed_frame_id
        dx_min, dy_min, dx_max, dy_max = self.velocity
        return BoundingBox(
            x_min=self.last_known_box.x_min + elapsed * dx_min,
            y_min=self.last_known_box.y_min + elapsed * dy_min,
            x_max=self.last_known_box.x_max + elapsed * dx_max,
            y_max=self.last_known_box.y_max + elapsed * dy_max,
        )


class SpatialReidentifier:
    """Retention pool and spatial re-identification matcher for retired entities.

    Maintains a retention pool bounded by ``retention_window`` frames. When a new
    unmatched detection is observed, ``match`` evaluates spatial plausibility against
    retained candidates based on predicted trajectory.

    Disambiguation Rule:
    When a new detection is spatially plausible against MORE THAN ONE retained candidate,
    match the candidate with the smallest prediction error (Euclidean distance between the
    detection box center and the candidate's predicted box center). In the case of an exact
    prediction error tie, the match resolves to the candidate with the lowest ``entity_id``
    (creation order precedent).
    """

    def __init__(
        self,
        retention_window: int = 10,
        max_distance: float | None = 50.0,
        min_iou: float | None = None,
    ) -> None:
        if retention_window < 0:
            raise ValueError(
                f"retention_window ({retention_window}) must be non-negative"
            )
        if max_distance is None and min_iou is None:
            raise ValueError("At least one of max_distance or min_iou must be specified")
        if max_distance is not None and max_distance < 0.0:
            raise ValueError(f"max_distance ({max_distance}) must be >= 0.0")
        if min_iou is not None and not (0.0 <= min_iou <= 1.0):
            raise ValueError(f"min_iou ({min_iou}) must be within [0.0, 1.0]")

        self._retention_window = retention_window
        self._max_distance = max_distance
        self._min_iou = min_iou
        self._candidates: list[ReidentificationCandidate] = []

    @property
    def retention_window(self) -> int:
        return self._retention_window

    @property
    def candidates(self) -> list[ReidentificationCandidate]:
        return list(self._candidates)

    def add_candidate(self, candidate: ReidentificationCandidate) -> None:
        """Add a retired entity candidate to the retention pool."""
        self._candidates.append(candidate)

    def purge_expired(self, current_frame_id: int) -> None:
        """Permanently purge candidates exceeding retention_window."""
        self._candidates = [
            c
            for c in self._candidates
            if current_frame_id - c.retired_frame_id <= self._retention_window
        ]

    def match(
        self, detection: Detection, frame_id: int
    ) -> ReidentificationCandidate | None:
        """Match ``detection`` against retained candidates at ``frame_id``.

        Returns the best matching candidate (and removes it from pool) or ``None``.
        """
        self.purge_expired(frame_id)
        if not self._candidates:
            return None

        det_box = detection.bounding_box
        det_cx = (det_box.x_min + det_box.x_max) / 2.0
        det_cy = (det_box.y_min + det_box.y_max) / 2.0

        plausible: list[tuple[float, int, ReidentificationCandidate]] = []

        for cand in self._candidates:
            if cand.class_label != detection.class_label:
                continue

            pred_box = cand.predict_box(frame_id)
            pred_cx = (pred_box.x_min + pred_box.x_max) / 2.0
            pred_cy = (pred_box.y_min + pred_box.y_max) / 2.0
            dist = math.hypot(det_cx - pred_cx, det_cy - pred_cy)
            overlap = iou(det_box, pred_box)

            if self._max_distance is not None and dist > self._max_distance:
                continue
            if self._min_iou is not None and overlap < self._min_iou:
                continue

            plausible.append((dist, cand.entity_id, cand))

        if not plausible:
            return None

        # Disambiguation: smallest prediction error (dist) first, tie-break lowest entity_id
        plausible.sort(key=lambda item: (item[0], item[1]))
        best_candidate = plausible[0][2]
        self._candidates.remove(best_candidate)
        return best_candidate
