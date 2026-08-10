"""Thresholded precision/recall for per-frame detection evaluation."""

from __future__ import annotations

from sentinel_vision.data.contracts import Detection, GroundTruthAnnotation
from sentinel_vision.evaluation.geometry import iou


def precision_recall_at_iou(
    predictions: list[Detection],
    ground_truth: GroundTruthAnnotation,
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """Greedy-matching precision and recall at a fixed IoU threshold.

    Predictions are sorted by confidence descending. A prediction may match
    at most one still-unmatched ground-truth box, and only when BOTH hold:
    its IoU with the box is at least ``iou_threshold`` AND its
    ``class_label`` equals the ground-truth box's label (exact, case
    sensitive string match). A prediction overlapping a box of a different
    class does not match it — it counts as a false positive, and that
    ground-truth box remains a false negative.

    Matching is one-to-one: once a ground-truth box is matched it is
    removed from consideration. Among the ground-truth boxes a prediction
    qualifies for, the box with the strictly highest IoU wins; ties are
    broken by strict ``>`` comparison, so the first box encountered (the
    lowest index) wins — iteration order is deterministic. Unmatched
    predictions are false positives; unmatched ground-truth boxes are
    false negatives.

    Edge cases (documented convention, never a division by zero):

    - ``predictions`` empty: ``(0.0, 0.0)`` — precision is defined as 0.0
      (not 1.0) to penalize inactivity: an idle detector contributes no
      positive claims and, in industrial monitoring, missing a frame's
      objects is a real cost we choose not to paper over. Recall is 0.0
      because no ground-truth box is matched.
    - ``ground_truth`` empty (no boxes): ``(0.0, 0.0)`` — every predicted
      box is spurious, so precision is 0.0 and there is no ground truth to
      recall.
    - both empty: ``(0.0, 0.0)`` — consistent with the two cases above; an
      evaluation over empty inputs reports no success rather than erroring.
    """

    matched_gt: set[int] = set()
    true_positives = 0

    for detection in sorted(predictions, key=lambda d: d.confidence, reverse=True):
        best_gt_index = -1
        best_iou = 0.0
        for gt_index, (gt_box, gt_label) in enumerate(
            zip(ground_truth.boxes, ground_truth.labels, strict=True)
        ):
            if gt_index in matched_gt:
                continue
            if detection.class_label != gt_label:
                continue
            candidate = iou(detection.bounding_box, gt_box)
            if candidate < iou_threshold:
                continue
            if candidate > best_iou:
                best_iou = candidate
                best_gt_index = gt_index
        if best_gt_index >= 0:
            true_positives += 1
            matched_gt.add(best_gt_index)

    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth.boxes) - true_positives

    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    return (precision, recall)
