"""Tests for the synthetic threshold detector (PR-004).

Covers the empty-frame contract (zero detections, never an error), a
hand-computed single-object box, the fixed confidence and class label, the
enclosing-box simplification for disjoint regions, and the guarantee that
detecting never mutates the input frame.
"""

import numpy as np

from sentinel_vision.data.contracts import BoundingBox
from sentinel_vision.detection.synthetic import (
    DEFAULT_CONFIDENCE,
    SYNTHETIC_TARGET_LABEL,
    SyntheticBoxDetector,
)
from sentinel_vision.ingestion.contracts import FrameData


def make_frame(image: np.ndarray) -> FrameData:
    return FrameData(frame_id=0, timestamp_ms=0.0, image=image)


def make_blank_image(height: int = 10, width: int = 12) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestSyntheticBoxDetector:
    def test_empty_frame_yields_zero_detections(self) -> None:
        assert SyntheticBoxDetector().detect(make_frame(make_blank_image())) == []

    def test_single_object_yields_one_detection_with_hand_computed_box(self) -> None:
        # A 3x4 block at rows 2..4, cols 5..8 on an otherwise black frame.
        # Exclusive-max box: (x_min=5, y_min=2, x_max=9, y_max=5).
        image = make_blank_image()
        image[2:5, 5:9] = 255
        detections = SyntheticBoxDetector().detect(make_frame(image))
        assert len(detections) == 1
        assert detections[0].bounding_box == BoundingBox(5.0, 2.0, 9.0, 5.0)

    def test_confidence_is_always_one(self) -> None:
        image = make_blank_image()
        image[0:2, 0:2] = 255
        detections = SyntheticBoxDetector().detect(make_frame(image))
        assert len(detections) == 1
        assert detections[0].confidence == 1.0
        assert detections[0].confidence == DEFAULT_CONFIDENCE

    def test_class_label_is_always_the_fixed_constant(self) -> None:
        image = make_blank_image()
        image[0:2, 0:2] = 255
        detections = SyntheticBoxDetector().detect(make_frame(image))
        assert detections[0].class_label == "synthetic_target"
        assert detections[0].class_label == SYNTHETIC_TARGET_LABEL

    def test_disjoint_regions_are_enclosed_in_one_box(self) -> None:
        # Two separate blocks: (rows 1, cols 2) and (rows 6..7, cols 7..8).
        # One enclosing box spans the union: (x_min=2, y_min=1, x_max=9, y_max=8).
        image = make_blank_image()
        image[1:2, 2:3] = 255
        image[6:8, 7:9] = 255
        detections = SyntheticBoxDetector().detect(make_frame(image))
        assert len(detections) == 1
        assert detections[0].bounding_box == BoundingBox(2.0, 1.0, 9.0, 8.0)

    def test_detect_does_not_mutate_input_frame(self) -> None:
        image = make_blank_image()
        image[2:5, 5:9] = 255
        frame = make_frame(image)
        original = frame.image.copy()
        SyntheticBoxDetector().detect(frame)
        assert np.array_equal(frame.image, original)
        assert not frame.image.flags.writeable
