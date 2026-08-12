"""MOTA and IDF1 tracking evaluation for the deterministic pipeline (PR-005).

Both metrics share the same per-frame matching: predicted ``TrackedDetection``
objects are greedily matched to ground-truth boxes by IoU at
``iou_threshold``, requiring exact, case-sensitive class-label equality
(mirroring ``precision_recall_at_iou`` from PR-002).

MOTA is a per-frame detection-level score: ``1 - (FN + FP + IDSW) / GT``,
where an identity switch is a ground-truth track reassigned to a different
predicted track id than the one it last matched. IDF1 is a track-level
identity score: per-frame matches are counted per track pair, a global
one-to-one assignment between ground-truth and predicted tracks maximizes
the total matched detections, and ``IDF1 = 2*IDTP / (2*IDTP + IDFN + IDFP)``
(see docs/adr/0005-tracker-and-evaluation-harness.md).
"""

from __future__ import annotations

from collections.abc import Sequence

from sentinel_vision.data.contracts import GroundTruthAnnotation, TrackedDetection
from sentinel_vision.evaluation.geometry import iou

TrackKey = int | tuple[int, int, int]


def _validate_frame_count(
    ground_truth_tracks: Sequence[GroundTruthAnnotation],
    predicted_tracks: Sequence[Sequence[TrackedDetection]],
) -> None:
    if len(ground_truth_tracks) != len(predicted_tracks):
        raise ValueError(
            f"len(ground_truth_tracks)={len(ground_truth_tracks)} != "
            f"len(predicted_tracks)={len(predicted_tracks)}"
        )


def _frame_matches(
    ground_truth: GroundTruthAnnotation,
    predictions: Sequence[TrackedDetection],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """Greedily match one frame's predictions to its ground truth.

    Predictions are sorted by confidence descending (stable, so ties keep
    input order); each takes the best still-unmatched ground-truth box with
    the same class label and IoU at least ``iou_threshold``. Returns the
    matched pairs as ``(ground_truth_index, prediction_index)`` where the
    prediction index is the position in the original ``predictions`` list.
    """
    matched_gt: set[int] = set()
    matches: list[tuple[int, int]] = []
    for p_idx, prediction in sorted(
        enumerate(predictions),
        key=lambda item: item[1].detection.confidence,
        reverse=True,
    ):
        best_gt = -1
        best_iou = 0.0
        for g_idx, (gt_box, gt_label) in enumerate(
            zip(ground_truth.boxes, ground_truth.labels, strict=True)
        ):
            if g_idx in matched_gt:
                continue
            if prediction.detection.class_label != gt_label:
                continue
            candidate = iou(prediction.detection.bounding_box, gt_box)
            if candidate < iou_threshold:
                continue
            if candidate > best_iou:
                best_iou = candidate
                best_gt = g_idx
        if best_gt >= 0:
            matched_gt.add(best_gt)
            matches.append((best_gt, p_idx))
    return matches


def _gt_track_key(ground_truth: GroundTruthAnnotation, box_index: int) -> TrackKey:
    """Identity key for a ground-truth box's track.

    A non-negative ``track_id`` keys the track directly. A ``-1`` track id
    (no identity recorded, per PR-002's contracts) keys the box as its own
    one-frame track, so it never accumulates identity across frames.
    """
    track_id = ground_truth.track_ids[box_index]
    if track_id >= 0:
        return track_id
    return (track_id, ground_truth.frame.frame_index, box_index)


def calculate_mota(
    ground_truth_tracks: Sequence[GroundTruthAnnotation],
    predicted_tracks: Sequence[Sequence[TrackedDetection]],
    iou_threshold: float = 0.5,
) -> float:
    """Multiple-object tracking accuracy over a frame sequence.

    ``ground_truth_tracks`` and ``predicted_tracks`` are parallel sequences
    indexed by frame: one ``GroundTruthAnnotation`` and one list of
    ``TrackedDetection`` per frame, in stream order.

    Identity switches are counted per ground-truth track: when a track is
    matched by a different predicted track id than the one it last matched.
    A track that goes unmatched keeps its previous assignment, so an object
    that reappears under a new predicted id is scored as a switch. MOTA is
    unbounded below: with ``FN + FP + IDSW > GT`` the score is negative,
    which is a valid signal of a degraded tracker, not an error.

    Documented edge cases (never a division by zero): both ground truth and
    predictions empty -> ``1.0``; ground truth empty with predictions ->
    ``0.0`` (every prediction is spurious); predictions empty with ground
    truth -> ``0.0`` (every object is missed).
    """
    _validate_frame_count(ground_truth_tracks, predicted_tracks)
    total_gt = sum(len(f.boxes) for f in ground_truth_tracks)
    total_pred = sum(len(p) for p in predicted_tracks)
    if total_gt == 0:
        return 1.0 if total_pred == 0 else 0.0

    fn = 0
    fp = 0
    idsw = 0
    prev_assignment: dict[TrackKey, int] = {}
    for ground_truth, preds in zip(
        ground_truth_tracks, predicted_tracks, strict=True
    ):
        matches = _frame_matches(ground_truth, preds, iou_threshold)
        fn += len(ground_truth.boxes) - len(matches)
        fp += len(preds) - len(matches)
        current_assignment = dict(prev_assignment)
        for g_idx, p_idx in matches:
            gt_key = _gt_track_key(ground_truth, g_idx)
            pred_track = preds[p_idx].track_id
            if gt_key in prev_assignment and prev_assignment[gt_key] != pred_track:
                idsw += 1
            current_assignment[gt_key] = pred_track
        prev_assignment = current_assignment

    return 1.0 - (fn + fp + idsw) / total_gt


def _maximize_assignment(scores: list[list[float]]) -> list[int]:
    """Maximize total score over a one-to-one track assignment.

    Rows are ground-truth tracks, columns predicted tracks; ``scores[i][j]``
    is the number of frames track pair ``(i, j)`` was matched. Returns for
    each row the assigned column, or ``-1`` when left unmatched. The optimal
    assignment is found with a pure-Python Hungarian algorithm (the harness
    stays NumPy/standard-library-only per ADR-0005).
    """
    n = len(scores)
    if n == 0:
        return []
    m = len(scores[0])
    if m == 0:
        return [-1] * n

    size = max(n, m)
    cost = [[0.0] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            cost[i][j] = -scores[i][j]

    assignment = _hungarian_min(cost)
    return [assignment[i] if assignment[i] < m else -1 for i in range(n)]


def _hungarian_min(cost: list[list[float]]) -> list[int]:
    """Minimum-cost perfect matching on a square matrix (Kuhn-Munkres).

    Returns for each row the assigned column index. Handles negative costs,
    which the dummy-row/column padding in ``_maximize_assignment`` relies on.
    """
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def calculate_idf1(
    ground_truth_tracks: Sequence[GroundTruthAnnotation],
    predicted_tracks: Sequence[Sequence[TrackedDetection]],
    iou_threshold: float = 0.5,
) -> float:
    """Identification F1 over a frame sequence (MOTChallenge formulation).

    Per-frame matches are counted per (ground-truth track, predicted track)
    pair, then a global one-to-one assignment maximizes the total matched
    detections ``IDTP``. ``IDFN`` is the ground-truth detections left
    unmatched by the assignment and ``IDFP`` the spurious predicted
    detections; ``IDF1 = 2*IDTP / (2*IDTP + IDFN + IDFP)``.

    Documented edge cases: both ground truth and predictions empty -> ``1.0``;
    ground truth empty with predictions -> ``0.0``; predictions empty with
    ground truth -> ``0.0``.
    """
    _validate_frame_count(ground_truth_tracks, predicted_tracks)
    total_gt = sum(len(f.boxes) for f in ground_truth_tracks)
    total_pred = sum(len(p) for p in predicted_tracks)
    if total_gt == 0:
        return 1.0 if total_pred == 0 else 0.0

    gt_keys: set[TrackKey] = set()
    pred_tracks: set[int] = set()
    pair_counts: dict[tuple[TrackKey, int], int] = {}
    for ground_truth, preds in zip(
        ground_truth_tracks, predicted_tracks, strict=True
    ):
        for g_idx in range(len(ground_truth.boxes)):
            gt_keys.add(_gt_track_key(ground_truth, g_idx))
        for prediction in preds:
            pred_tracks.add(prediction.track_id)
        for g_idx, p_idx in _frame_matches(ground_truth, preds, iou_threshold):
            key = (_gt_track_key(ground_truth, g_idx), preds[p_idx].track_id)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    gt_track_list = sorted(gt_keys, key=repr)
    pred_track_list = sorted(pred_tracks)
    scores = [[0.0] * len(pred_track_list) for _ in gt_track_list]
    gt_index = {key: i for i, key in enumerate(gt_track_list)}
    pred_index = {key: j for j, key in enumerate(pred_track_list)}
    for (gt_key, pred_track), count in pair_counts.items():
        scores[gt_index[gt_key]][pred_index[pred_track]] = float(count)

    assignment = _maximize_assignment(scores)
    idtp = 0.0
    for row, col in enumerate(assignment):
        if col >= 0:
            idtp += scores[row][col]

    idfn = total_gt - idtp
    idfp = total_pred - idtp
    return 2.0 * idtp / (2.0 * idtp + idfn + idfp)
