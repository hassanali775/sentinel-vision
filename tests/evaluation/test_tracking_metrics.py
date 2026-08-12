"""Tests for MOTA and IDF1 tracking metrics (PR-005).

Covers perfect tracking, both-empty / zero-ground-truth conventions,
hand-computed false negatives and false positives, identity switches, the
global assignment IDF1 performs over a multi-track trajectory, and
class-aware matching.
"""

import pytest

from sentinel_vision.data.contracts import (
    BoundingBox,
    Detection,
    FrameRef,
    GroundTruthAnnotation,
    TrackedDetection,
)
from sentinel_vision.evaluation.tracking_metrics import calculate_idf1, calculate_mota


def box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def det(box_coords: tuple[float, float, float, float]) -> Detection:
    return Detection(
        bounding_box=box(*box_coords),
        confidence=1.0,
        class_label="synthetic_target",
    )


def tracked(
    box_coords: tuple[float, float, float, float], track_id: int
) -> TrackedDetection:
    return TrackedDetection(detection=det(box_coords), track_id=track_id)


def gt(
    frame_index: int,
    boxes: list[tuple[float, float, float, float]],
    track_ids: list[int],
    labels: list[str] | None = None,
) -> GroundTruthAnnotation:
    return GroundTruthAnnotation(
        frame=FrameRef(
            source_id="clip",
            frame_index=frame_index,
            timestamp=float(frame_index),
        ),
        boxes=[box(*coords) for coords in boxes],
        labels=["synthetic_target"] * len(boxes) if labels is None else labels,
        track_ids=list(track_ids),
    )


class TestCalculateMota:
    def test_perfect_tracking_is_one(self) -> None:
        ground_truth = [
            gt(0, [(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1]),
            gt(1, [(5, 0, 15, 10), (25, 20, 35, 30)], [0, 1]),
        ]
        predictions = [
            [tracked((0, 0, 10, 10), 0), tracked((20, 20, 30, 30), 1)],
            [tracked((5, 0, 15, 10), 0), tracked((25, 20, 35, 30), 1)],
        ]
        assert calculate_mota(ground_truth, predictions) == pytest.approx(1.0)

    def test_both_empty_is_one(self) -> None:
        assert calculate_mota([gt(0, [], [])], [[]]) == 1.0

    def test_ground_truth_empty_with_predictions_is_zero(self) -> None:
        ground_truth = [gt(0, [], [])]
        predictions = [[tracked((0, 0, 10, 10), 0)]]
        assert calculate_mota(ground_truth, predictions) == 0.0

    def test_all_false_negatives_is_zero(self) -> None:
        ground_truth = [gt(0, [(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1])]
        predictions: list[list[TrackedDetection]] = [[]]
        assert calculate_mota(ground_truth, predictions) == 0.0

    def test_hand_computed_missing_detection(self) -> None:
        # One object over two frames; the tracker misses frame 1.
        # GT=2, FN=1, FP=0, IDSW=0 -> MOTA = 1 - 1/2 = 0.5.
        ground_truth = [
            gt(0, [(0, 0, 10, 10)], [0]),
            gt(1, [(0, 0, 10, 10)], [0]),
        ]
        predictions = [
            [tracked((0, 0, 10, 10), 5)],
            [],
        ]
        assert calculate_mota(ground_truth, predictions) == pytest.approx(0.5)

    def test_hand_computed_identity_switch(self) -> None:
        # Two objects over two frames; the predicted track ids swap on the
        # second frame. Every detection matches, so FN=FP=0, but each ground
        # truth track is reassigned -> IDSW=2.
        # GT=4, IDSW=2 -> MOTA = 1 - 2/4 = 0.5.
        ground_truth = [
            gt(0, [(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1]),
            gt(1, [(5, 0, 15, 10), (25, 20, 35, 30)], [0, 1]),
        ]
        predictions = [
            [tracked((0, 0, 10, 10), 0), tracked((20, 20, 30, 30), 1)],
            [tracked((5, 0, 15, 10), 1), tracked((25, 20, 35, 30), 0)],
        ]
        assert calculate_mota(ground_truth, predictions) == pytest.approx(0.5)

    def test_identity_switch_across_gap_counts_once(self) -> None:
        # One object present in frames 0 and 2, absent in frame 1. The
        # predicted id changes across the gap: the carried-over assignment
        # makes that exactly one IDSW. GT=2, IDSW=1 -> MOTA = 1 - 1/2 = 0.5.
        ground_truth = [
            gt(0, [(0, 0, 10, 10)], [0]),
            gt(1, [], []),
            gt(2, [(0, 0, 10, 10)], [0]),
        ]
        predictions = [
            [tracked((0, 0, 10, 10), 3)],
            [],
            [tracked((0, 0, 10, 10), 4)],
        ]
        assert calculate_mota(ground_truth, predictions) == pytest.approx(0.5)

    def test_class_mismatch_counts_as_fn_and_fp(self) -> None:
        # A prediction of class "person" perfectly overlapping a GT box
        # labeled "synthetic_target" does not match: FN=1 and FP=1 over one
        # GT detection -> MOTA = 1 - 2/1 = -1.0 (valid, unbounded below).
        ground_truth = [gt(0, [(0, 0, 10, 10)], [0], labels=["synthetic_target"])]
        prediction = TrackedDetection(
            detection=Detection(
                bounding_box=box(0, 0, 10, 10),
                confidence=1.0,
                class_label="person",
            ),
            track_id=0,
        )
        assert calculate_mota(ground_truth, [[prediction]]) == pytest.approx(-1.0)

    def test_mismatched_frame_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="ground_truth_tracks"):
            calculate_mota([gt(0, [], [])], [])


class TestCalculateIdf1:
    def test_perfect_tracking_is_one(self) -> None:
        ground_truth = [
            gt(0, [(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1]),
            gt(1, [(5, 0, 15, 10), (25, 20, 35, 30)], [0, 1]),
        ]
        predictions = [
            [tracked((0, 0, 10, 10), 0), tracked((20, 20, 30, 30), 1)],
            [tracked((5, 0, 15, 10), 0), tracked((25, 20, 35, 30), 1)],
        ]
        assert calculate_idf1(ground_truth, predictions) == pytest.approx(1.0)

    def test_both_empty_is_one(self) -> None:
        assert calculate_idf1([gt(0, [], [])], [[]]) == 1.0

    def test_ground_truth_empty_with_predictions_is_zero(self) -> None:
        assert calculate_idf1([gt(0, [], [])], [[tracked((0, 0, 10, 10), 0)]]) == 0.0

    def test_all_false_negatives_is_zero(self) -> None:
        ground_truth = [gt(0, [(0, 0, 10, 10)], [0])]
        predictions: list[list[TrackedDetection]] = [[]]
        assert calculate_idf1(ground_truth, predictions) == 0.0

    def test_hand_computed_missing_detection(self) -> None:
        # Two GT detections, one matched: IDTP=1, IDFN=1, IDFP=0
        # -> IDF1 = 2/(2+1+0) = 2/3.
        ground_truth = [
            gt(0, [(0, 0, 10, 10)], [0]),
            gt(1, [(0, 0, 10, 10)], [0]),
        ]
        predictions = [
            [tracked((0, 0, 10, 10), 5)],
            [],
        ]
        assert calculate_idf1(ground_truth, predictions) == pytest.approx(2 / 3)

    def test_hand_computed_identity_switch(self) -> None:
        # Two tracks over two frames with the ids swapped on frame 1: each
        # track pair matches one frame, so the best assignment scores 2.
        # IDTP=2, IDFN=2, IDFP=2 -> IDF1 = 4/(4+2+2) = 0.5.
        ground_truth = [
            gt(0, [(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1]),
            gt(1, [(5, 0, 15, 10), (25, 20, 35, 30)], [0, 1]),
        ]
        predictions = [
            [tracked((0, 0, 10, 10), 0), tracked((20, 20, 30, 30), 1)],
            [tracked((5, 0, 15, 10), 1), tracked((25, 20, 35, 30), 0)],
        ]
        assert calculate_idf1(ground_truth, predictions) == pytest.approx(0.5)

    def test_hand_computed_global_assignment(self) -> None:
        # Two tracks over three frames, ids swapped only on the last frame:
        # gt0->pred0 and gt1->pred1 each match two frames. IDTP=4, IDFN=2,
        # IDFP=2 -> IDF1 = 8/(8+2+2) = 2/3.
        ground_truth = [
            gt(0, [(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1]),
            gt(1, [(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1]),
            gt(2, [(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1]),
        ]
        predictions = [
            [tracked((0, 0, 10, 10), 0), tracked((20, 20, 30, 30), 1)],
            [tracked((0, 0, 10, 10), 0), tracked((20, 20, 30, 30), 1)],
            [tracked((0, 0, 10, 10), 1), tracked((20, 20, 30, 30), 0)],
        ]
        assert calculate_idf1(ground_truth, predictions) == pytest.approx(2 / 3)

    def test_mismatched_frame_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="ground_truth_tracks"):
            calculate_idf1([gt(0, [], [])], [])
