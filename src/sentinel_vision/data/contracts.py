"""Frozen data contracts shared across the deterministic pipeline.

These dataclasses are the vocabulary between dataset curation, the detection
abstraction (PR-004), the tracking layer (PR-005), and the evaluation layer.
Every one of them describes a fact about a specific frame — what was
annotated there, what was predicted there — so they are frozen: nothing in
the pipeline is allowed to mutate a frame's facts after the fact.

Validation happens in ``__post_init__`` for the fields where a malformed
value is a data-quality bug, not a value to silently accept: a degenerate
box, a confidence outside [0.0, 1.0], or a ground-truth annotation whose
box/label/track-id lists disagree in length.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned box in pixel coordinates.

    The convention is inclusive of ``x_min``/``y_min`` and exclusive of
    ``x_max``/``y_max`` so that a box's width/height are simple differences
    and edge-touching boxes never overlap.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(
                f"degenerate box: x_min ({self.x_min}) must be < x_max ({self.x_max})"
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"degenerate box: y_min ({self.y_min}) must be < y_max ({self.y_max})"
            )


@dataclass(frozen=True)
class Detection:
    """A single predicted box produced by a detector, with a confidence."""

    bounding_box: BoundingBox
    confidence: float
    class_label: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence ({self.confidence}) must be within [0.0, 1.0]"
            )


@dataclass(frozen=True)
class TrackedDetection:
    """A detection associated with a persistent track across frames.

    Produced by the tracking layer (PR-005): the tracker consumes
    ``Detection`` objects and tags each with the stable ``track_id`` of the
    object it was associated with. A ``-1`` track id is never valid here —
    unlike ``GroundTruthAnnotation``, which uses ``-1`` to mean "no identity
    recorded", a tracked detection always carries an identity.
    """

    detection: Detection
    track_id: int

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"track_id ({self.track_id}) must be non-negative")


@dataclass(frozen=True)
class FrameRef:
    """Identifies which frame any other object refers to."""

    source_id: str
    frame_index: int
    timestamp: float


@dataclass(frozen=True)
class GroundTruthAnnotation:
    """Hand-labeled ground truth for a single frame.

    ``boxes``, ``labels``, and ``track_ids`` are parallel lists: element
    ``i`` of each describes the same object. A ground-truth box may carry a
    ``-1`` track id when the annotator recorded no identity for it (e.g. a
    static object); track-based evaluation (PR-005/PR-006) only consumes
    annotations whose track ids are present.
    """

    frame: FrameRef
    boxes: list[BoundingBox]
    labels: list[str]
    track_ids: list[int]

    def __post_init__(self) -> None:
        box_count = len(self.boxes)
        if len(self.labels) != box_count:
            raise ValueError(
                f"len(boxes)={box_count} != len(labels)={len(self.labels)}"
            )
        if len(self.track_ids) != box_count:
            raise ValueError(
                f"len(boxes)={box_count} != len(track_ids)={len(self.track_ids)}"
            )
