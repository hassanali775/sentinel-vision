"""Tests for the frozen data contracts (PR-002, Phase 1).

Every validation rule must be proven to actually reject bad input: a
degenerate box, an out-of-range confidence, and mismatched list lengths.
"""

from dataclasses import FrozenInstanceError

import pytest

from sentinel_vision.data.contracts import (
    BoundingBox,
    Detection,
    FrameRef,
    GroundTruthAnnotation,
)


def make_box(
    x_min: float = 0.0, y_min: float = 0.0, x_max: float = 10.0, y_max: float = 10.0
) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def make_frame() -> FrameRef:
    return FrameRef(source_id="clip-a", frame_index=1, timestamp=1723000000.0)


def make_detection(confidence: float) -> Detection:
    return Detection(bounding_box=make_box(), confidence=confidence, class_label="part")


def make_annotation(
    boxes: list[BoundingBox], labels: list[str], track_ids: list[int]
) -> GroundTruthAnnotation:
    return GroundTruthAnnotation(
        frame=make_frame(), boxes=boxes, labels=labels, track_ids=track_ids
    )


class TestBoundingBox:
    def test_valid_box_constructs(self) -> None:
        box = make_box(x_max=10.0, y_max=20.0)
        assert (box.x_min, box.y_min, box.x_max, box.y_max) == (0.0, 0.0, 10.0, 20.0)

    def test_rejects_zero_width(self) -> None:
        with pytest.raises(ValueError, match="x_min"):
            make_box(x_min=10.0, x_max=10.0)

    def test_rejects_inverted_width(self) -> None:
        with pytest.raises(ValueError, match="x_min"):
            make_box(x_min=10.0, x_max=5.0)

    def test_rejects_zero_height(self) -> None:
        with pytest.raises(ValueError, match="y_min"):
            make_box(y_min=10.0, y_max=10.0)

    def test_rejects_inverted_height(self) -> None:
        with pytest.raises(ValueError, match="y_min"):
            make_box(y_min=20.0, y_max=5.0)


class TestDetection:
    def test_valid_detection_constructs(self) -> None:
        detection = make_detection(confidence=0.75)
        assert detection.class_label == "part"
        assert detection.confidence == 0.75

    def test_confidence_boundaries_are_valid(self) -> None:
        for confidence in (0.0, 1.0):
            make_detection(confidence=confidence)

    def test_rejects_negative_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            make_detection(confidence=-0.01)

    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            make_detection(confidence=1.01)


class TestFrameRef:
    def test_valid_frame_ref_constructs(self) -> None:
        frame = make_frame()
        assert (frame.source_id, frame.frame_index, frame.timestamp) == ("clip-a", 1, 1723000000.0)


class TestGroundTruthAnnotation:
    def test_valid_annotation_constructs(self) -> None:
        boxes = [make_box(), make_box(x_min=20.0, x_max=30.0)]
        make_annotation(boxes=boxes, labels=["part", "part"], track_ids=[1, 2])

    def test_rejects_mismatched_labels(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            make_annotation(boxes=[make_box()], labels=["part", "part"], track_ids=[1])

    def test_rejects_mismatched_track_ids(self) -> None:
        boxes = [make_box(), make_box(x_min=20.0, x_max=30.0)]
        with pytest.raises(ValueError, match="track_ids"):
            make_annotation(boxes=boxes, labels=["part", "part"], track_ids=[1])

    def test_annotation_with_no_objects_is_valid(self) -> None:
        make_annotation(boxes=[], labels=[], track_ids=[])


class TestImmutability:
    def test_contracts_are_frozen(self) -> None:
        box = make_box()
        with pytest.raises(FrozenInstanceError):
            box.x_min = 1.0  # type: ignore[misc]
