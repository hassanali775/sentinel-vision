"""Tests for the greedy IoU tracker (PR-005).

Covers track creation, stable track ids across linear movement, distinct
ids for multiple objects, track survival and expiration under ``max_age``,
the inclusive ``iou_threshold`` semantics, and constructor validation.
"""

import numpy as np
import pytest

from sentinel_vision.data.contracts import BoundingBox, Detection, TrackedDetection
from sentinel_vision.ingestion.contracts import FrameData
from sentinel_vision.tracking.greedy import GreedyIoUTracker


def make_frame(frame_id: int = 0) -> FrameData:
    return FrameData(
        frame_id=frame_id,
        timestamp_ms=float(frame_id),
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def det(
    x_min: float, y_min: float, x_max: float, y_max: float, confidence: float = 1.0
) -> Detection:
    return Detection(
        bounding_box=box(x_min, y_min, x_max, y_max),
        confidence=confidence,
        class_label="synthetic_target",
    )


def track_ids(results: list[TrackedDetection]) -> list[int]:
    return [result.track_id for result in results]


class TestGreedyIoUTracker:
    def test_first_detection_creates_new_track(self) -> None:
        results = GreedyIoUTracker().track(make_frame(), [det(0, 0, 10, 10)])
        assert track_ids(results) == [0]

    def test_matching_detection_keeps_track_id(self) -> None:
        tracker = GreedyIoUTracker()
        first = tracker.track(make_frame(0), [det(0, 0, 10, 10)])
        second = tracker.track(make_frame(1), [det(1, 0, 11, 10)])
        assert track_ids(first) == [0]
        assert track_ids(second) == [0]

    def test_stable_id_across_linear_movement(self) -> None:
        # A 10x10 box moving 1px per frame overlaps each predecessor by
        # IoU ~0.82 (> 0.5), so the tracker keeps one track across the run.
        tracker = GreedyIoUTracker()
        ids = []
        for frame_id in range(5):
            result = tracker.track(make_frame(frame_id), [det(frame_id, 0, frame_id + 10, 10)])
            ids.append(result[0].track_id)
        assert ids == [0, 0, 0, 0, 0]

    def test_unmatched_detection_creates_new_track(self) -> None:
        tracker = GreedyIoUTracker()
        first = tracker.track(make_frame(0), [det(0, 0, 10, 10)])
        second = tracker.track(make_frame(1), [det(50, 50, 60, 60)])
        assert track_ids(first) == [0]
        assert track_ids(second) == [1]

    def test_multiple_objects_keep_distinct_stable_ids(self) -> None:
        tracker = GreedyIoUTracker()
        frame0 = tracker.track(make_frame(0), [det(0, 0, 10, 10), det(30, 0, 40, 10)])
        frame1 = tracker.track(make_frame(1), [det(1, 0, 11, 10), det(31, 0, 41, 10)])
        assert track_ids(frame0) == [0, 1]
        assert track_ids(frame1) == [0, 1]

    def test_confidence_order_prefers_high_confidence_match(self) -> None:
        # Two detections in one frame: the high-confidence one takes the
        # single overlapping track; the low-confidence one starts a new one.
        tracker = GreedyIoUTracker()
        frame0 = tracker.track(make_frame(0), [det(0, 0, 10, 10)])
        assert track_ids(frame0) == [0]
        frame1 = tracker.track(
            make_frame(1),
            [
                det(1, 0, 11, 10, confidence=0.9),
                det(0, 0, 10, 10, confidence=0.5),
            ],
        )
        assert track_ids(frame1) == [0, 1]

    def test_unmatched_track_survives_within_max_age(self) -> None:
        tracker = GreedyIoUTracker(max_age=1)
        assert track_ids(tracker.track(make_frame(0), [det(0, 0, 10, 10)])) == [0]
        assert tracker.track(make_frame(1), []) == []
        assert track_ids(tracker.track(make_frame(2), [det(0, 0, 10, 10)])) == [0]

    def test_track_is_dropped_after_exceeding_max_age(self) -> None:
        tracker = GreedyIoUTracker(max_age=1)
        assert track_ids(tracker.track(make_frame(0), [det(0, 0, 10, 10)])) == [0]
        assert tracker.track(make_frame(1), []) == []
        assert tracker.track(make_frame(2), []) == []
        assert track_ids(tracker.track(make_frame(3), [det(0, 0, 10, 10)])) == [1]

    def test_max_age_zero_drops_after_one_unmatched_frame(self) -> None:
        tracker = GreedyIoUTracker(max_age=0)
        assert track_ids(tracker.track(make_frame(0), [det(0, 0, 10, 10)])) == [0]
        assert tracker.track(make_frame(1), []) == []
        assert track_ids(tracker.track(make_frame(2), [det(0, 0, 10, 10)])) == [1]

    def test_detection_below_iou_threshold_creates_new_track(self) -> None:
        # IoU between (0,0,10,10) and (4,0,14,10) is 60/140 ~ 0.43 < 0.5.
        tracker = GreedyIoUTracker(iou_threshold=0.5)
        assert track_ids(tracker.track(make_frame(0), [det(0, 0, 10, 10)])) == [0]
        assert track_ids(tracker.track(make_frame(1), [det(4, 0, 14, 10)])) == [1]

    def test_detection_at_iou_threshold_matches(self) -> None:
        # IoU between (0,0,10,10) and (2,0,12,10) is 80/120 ~ 0.67 >= 0.5.
        tracker = GreedyIoUTracker(iou_threshold=0.5)
        assert track_ids(tracker.track(make_frame(0), [det(0, 0, 10, 10)])) == [0]
        assert track_ids(tracker.track(make_frame(1), [det(2, 0, 12, 10)])) == [0]

    def test_empty_detections_yields_empty(self) -> None:
        assert GreedyIoUTracker().track(make_frame(0), []) == []

    def test_rejects_iou_threshold_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="iou_threshold"):
            GreedyIoUTracker(iou_threshold=-0.1)
        with pytest.raises(ValueError, match="iou_threshold"):
            GreedyIoUTracker(iou_threshold=1.1)

    def test_rejects_negative_max_age(self) -> None:
        with pytest.raises(ValueError, match="max_age"):
            GreedyIoUTracker(max_age=-1)
