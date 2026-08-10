"""Tests for greedy-matching precision/recall (PR-002, Phase 3).

Covers perfect match, all-false-positive / all-false-negative / both-empty
edge cases, a hand-computed mixed case, the at-most-one-match rule, the
effect of the iou_threshold parameter, and class-aware matching (a
prediction only matches a ground-truth box of the same class).
"""

import pytest

from sentinel_vision.data.contracts import BoundingBox, Detection, FrameRef, GroundTruthAnnotation
from sentinel_vision.evaluation.detection_metrics import precision_recall_at_iou


def box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def det(
    box_coords: tuple[float, float, float, float], confidence: float, class_label: str = "part"
) -> Detection:
    return Detection(bounding_box=box(*box_coords), confidence=confidence, class_label=class_label)


def annotation(
    boxes: list[BoundingBox], labels: list[str], track_ids: list[int]
) -> GroundTruthAnnotation:
    return GroundTruthAnnotation(
        frame=FrameRef(source_id="clip-a", frame_index=1, timestamp=1723000000.0),
        boxes=boxes,
        labels=labels,
        track_ids=track_ids,
    )


class TestPrecisionRecallAtIoU:
    def test_perfect_match(self) -> None:
        predictions = [
            det((0.0, 0.0, 10.0, 10.0), 0.9),
            det((20.0, 20.0, 30.0, 30.0), 0.8),
        ]
        ground_truth = annotation(
            [box(0.0, 0.0, 10.0, 10.0), box(20.0, 20.0, 30.0, 30.0)],
            ["part", "part"],
            [1, 2],
        )
        assert precision_recall_at_iou(predictions, ground_truth) == pytest.approx((1.0, 1.0))

    def test_all_false_positives_when_ground_truth_empty(self) -> None:
        predictions = [
            det((0.0, 0.0, 10.0, 10.0), 0.9),
            det((50.0, 50.0, 60.0, 60.0), 0.3),
        ]
        assert precision_recall_at_iou(predictions, annotation([], [], [])) == (0.0, 0.0)

    def test_all_false_negatives_when_predictions_empty(self) -> None:
        ground_truth = annotation(
            [box(0.0, 0.0, 10.0, 10.0), box(20.0, 20.0, 30.0, 30.0)],
            ["part", "part"],
            [1, 2],
        )
        assert precision_recall_at_iou([], ground_truth) == (0.0, 0.0)

    def test_both_empty(self) -> None:
        assert precision_recall_at_iou([], annotation([], [], [])) == (0.0, 0.0)

    def test_mixed_case_hand_computed(self) -> None:
        # GT: a=(0,0,10,10), b=(20,20,30,30).
        # Preds: a-exact @0.9 -> TP (matches a), a-duplicate @0.6 -> FP (a already
        # matched, no overlap with b), far box @0.3 -> FP. b is never matched -> FN.
        # TP=1, FP=2, FN=1 -> precision 1/3, recall 1/2.
        predictions = [
            det((0.0, 0.0, 10.0, 10.0), 0.9),
            det((0.0, 0.0, 10.0, 10.0), 0.6),
            det((50.0, 50.0, 60.0, 60.0), 0.3),
        ]
        ground_truth = annotation(
            [box(0.0, 0.0, 10.0, 10.0), box(20.0, 20.0, 30.0, 30.0)],
            ["part", "part"],
            [1, 2],
        )
        precision, recall = precision_recall_at_iou(predictions, ground_truth)
        assert precision == pytest.approx(1 / 3)
        assert recall == pytest.approx(1 / 2)

    def test_ground_truth_box_matched_at_most_once(self) -> None:
        # Two predictions overlap the single GT box with IoU >= 0.5; only one
        # can match, so exactly one prediction is a true positive.
        predictions = [
            det((0.0, 0.0, 10.0, 10.0), 0.9),
            det((1.0, 1.0, 9.0, 9.0), 0.5),
        ]
        ground_truth = annotation([box(0.0, 0.0, 10.0, 10.0)], ["part"], [1])
        assert precision_recall_at_iou(predictions, ground_truth) == pytest.approx((0.5, 1.0))

    def test_lower_threshold_allows_weaker_overlap(self) -> None:
        # Prediction overlaps the GT box with IoU = 1/3: a miss at the default
        # 0.5 threshold, a hit once the threshold is lowered to 0.3.
        prediction = det((5.0, 0.0, 15.0, 10.0), 0.9)
        ground_truth = annotation([box(0.0, 0.0, 10.0, 10.0)], ["part"], [1])
        assert precision_recall_at_iou([prediction], ground_truth) == (0.0, 0.0)
        assert precision_recall_at_iou([prediction], ground_truth, iou_threshold=0.3) == (
            1.0,
            1.0,
        )

    def test_class_mismatch_is_false_positive(self) -> None:
        # A prediction of class "person" perfectly overlapping a GT box labeled
        # "forklift" must NOT match: IoU alone is not enough. The prediction is
        # a false positive and the GT box is a false negative -> (0.0, 0.0).
        predictions = [det((0.0, 0.0, 10.0, 10.0), 0.9, class_label="person")]
        ground_truth = annotation([box(0.0, 0.0, 10.0, 10.0)], ["forklift"], [1])
        assert precision_recall_at_iou(predictions, ground_truth) == (0.0, 0.0)

    def test_only_same_class_predictions_match(self) -> None:
        # Two predictions overlap the single "person" GT box; the "person"
        # prediction matches (TP) while the identical-geometry "forklift"
        # prediction does not (FP). Precision 1/2, recall 1.0.
        predictions = [
            det((0.0, 0.0, 10.0, 10.0), 0.9, class_label="person"),
            det((0.0, 0.0, 10.0, 10.0), 0.8, class_label="forklift"),
        ]
        ground_truth = annotation([box(0.0, 0.0, 10.0, 10.0)], ["person"], [1])
        assert precision_recall_at_iou(predictions, ground_truth) == pytest.approx((0.5, 1.0))

    def test_class_labels_are_case_sensitive(self) -> None:
        # "Person" != "person" at the exact string level, so no match.
        predictions = [det((0.0, 0.0, 10.0, 10.0), 0.9, class_label="Person")]
        ground_truth = annotation([box(0.0, 0.0, 10.0, 10.0)], ["person"], [1])
        assert precision_recall_at_iou(predictions, ground_truth) == (0.0, 0.0)
