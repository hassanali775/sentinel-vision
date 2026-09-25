"""Spatial/motion-only re-identification and candidate retention pool (PR-007).

This module implements ``ReidentificationCandidate`` and ``SpatialReidentifier``
to re-link newly detected tracks to recently retired entities based on trajectory
extrapolation (finite-difference velocity) without appearance features or heavy
dependencies (see docs/adr/0007-spatial-reidentification.md).

``plausibility_score`` is the single shared spatial plausibility rule of the
re-identification subsystem: it is used both by the RETIRED candidate pool in
``SpatialReidentifier.match`` and by the active (OCCLUDED/PREDICTED) entity pool
in ``PersistentEntityTracker`` so both tiers cannot drift apart in their
thresholds or tie-breaking (see docs/adr/0011-active-entity-spatial-matching.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sentinel_vision.data.contracts import BoundingBox, Detection
from sentinel_vision.evaluation.geometry import iou


def plausibility_score(
    detection_box: BoundingBox,
    detection_label: str,
    candidate_box: BoundingBox,
    candidate_label: str,
    max_distance: float | None,
    min_iou: float | None,
) -> float | None:
    """Score one detection/candidate pair as a spatially plausible match.

    This is the single shared plausibility rule of the re-identification
    subsystem. It was extracted verbatim from ``SpatialReidentifier.match`` so
    that the RETIRED candidate pool and the active (OCCLUDED/PREDICTED) entity
    pool are scored by exactly the same function with exactly the same
    thresholds (see docs/adr/0011-active-entity-spatial-matching.md). Two copies
    of a threshold rule is a class of divergence bug this project has hit before
    (PR-006/PR-007 velocity normalization), so the rule is written once.

    Returns:
        The Euclidean distance between the two box centers when the candidate is
        plausible, else ``None``. The distance doubles as the prediction-error
        ranking used to disambiguate between multiple plausible candidates
        (lowest wins), so no separate scoring function is needed.

    A candidate is plausible when:
    - the class labels are identical, and
    - ``max_distance`` is ``None`` or the center distance is within it, and
    - ``min_iou`` is ``None`` or the box overlap is at least it.

    Constraints are compared strictly: a distance exactly equal to
    ``max_distance`` and an IoU exactly equal to ``min_iou`` both remain
    plausible, preserving ``SpatialReidentifier.match``'s pre-extraction
    behavior byte for byte.
    """
    if candidate_label != detection_label:
        return None

    det_cx = (detection_box.x_min + detection_box.x_max) / 2.0
    det_cy = (detection_box.y_min + detection_box.y_max) / 2.0
    cand_cx = (candidate_box.x_min + candidate_box.x_max) / 2.0
    cand_cy = (candidate_box.y_min + candidate_box.y_max) / 2.0
    dist = math.hypot(det_cx - cand_cx, det_cy - cand_cy)
    overlap = iou(detection_box, candidate_box)

    if max_distance is not None and dist > max_distance:
        return None
    if min_iou is not None and overlap < min_iou:
        return None
    return dist


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
        """Linearly extrapolate bounding box to ``frame_id`` using velocity.

        The projection is clamped so the predicted box can never degenerate:
        if extrapolating a large frame gap amplifies a per-edge velocity
        difference enough to invert an axis, the box is re-centered on the
        extrapolated midpoint while keeping the last known size on that axis
        (observed on real video where detections arrive sparsely).
        """
        if frame_id < self.retired_frame_id:
            raise ValueError(
                f"frame_id ({frame_id}) cannot be before retired_frame_id ({self.retired_frame_id})"
            )
        elapsed = frame_id - self.last_observed_frame_id
        dx_min, dy_min, dx_max, dy_max = self.velocity
        last = self.last_known_box
        pred_x_min = last.x_min + elapsed * dx_min
        pred_x_max = last.x_max + elapsed * dx_max
        pred_y_min = last.y_min + elapsed * dy_min
        pred_y_max = last.y_max + elapsed * dy_max
        if pred_x_min >= pred_x_max:
            mid_x = (pred_x_min + pred_x_max) / 2.0
            half_width = (last.x_max - last.x_min) / 2.0
            pred_x_min, pred_x_max = mid_x - half_width, mid_x + half_width
        if pred_y_min >= pred_y_max:
            mid_y = (pred_y_min + pred_y_max) / 2.0
            half_height = (last.y_max - last.y_min) / 2.0
            pred_y_min, pred_y_max = mid_y - half_height, mid_y + half_height
        return BoundingBox(
            x_min=pred_x_min,
            y_min=pred_y_min,
            x_max=pred_x_max,
            y_max=pred_y_max,
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
    def max_distance(self) -> float | None:
        """Configured center-distance plausibility limit, or ``None`` if unset."""
        return self._max_distance

    @property
    def min_iou(self) -> float | None:
        """Configured minimum-overlap plausibility limit, or ``None`` if unset."""
        return self._min_iou

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

        The per-candidate plausibility test itself lives in the module-level
        ``plausibility_score``, which is shared with the active-entity pool in
        ``PersistentEntityTracker`` (ADR-0011). The candidate's box for
        ``frame_id`` is its linear extrapolation from ``last_known_box``;
        ``plausibility_score`` receives that already-extrapolated box.
        """
        self.purge_expired(frame_id)
        if not self._candidates:
            return None

        det_box = detection.bounding_box

        plausible: list[tuple[float, int, ReidentificationCandidate]] = []

        for cand in self._candidates:
            score = plausibility_score(
                det_box,
                detection.class_label,
                cand.predict_box(frame_id),
                cand.class_label,
                self._max_distance,
                self._min_iou,
            )
            if score is None:
                continue
            plausible.append((score, cand.entity_id, cand))

        if not plausible:
            return None

        # Disambiguation: smallest prediction error (dist) first, tie-break lowest entity_id
        plausible.sort(key=lambda item: (item[0], item[1]))
        best_candidate = plausible[0][2]
        self._candidates.remove(best_candidate)
        return best_candidate
