"""End-to-end detection + evaluation against the synthetic stream (PR-004).

Feeds every frame of a ``SyntheticFrameStream`` through
``SyntheticBoxDetector`` and scores the result with PR-002's
``precision_recall_at_iou``. The expected box for each frame is recomputed
in this test from the stream's public metadata — re-deriving the stream's
rendering formula rather than calling its private renderer or reading the
frame's pixels — so the precision/recall score measures the detector
against independent ground truth instead of being a tautology.
"""

import pytest

from sentinel_vision.data.contracts import BoundingBox, FrameRef, GroundTruthAnnotation
from sentinel_vision.detection.synthetic import SyntheticBoxDetector
from sentinel_vision.evaluation.detection_metrics import precision_recall_at_iou
from sentinel_vision.ingestion.stream import SyntheticFrameStream

GROUND_TRUTH_LABEL = "synthetic_target"


def expected_object_box(stream: SyntheticFrameStream, frame_id: int) -> BoundingBox:
    """Independently recompute the moving-object box for ``frame_id``.

    Re-derives the formula ``SyntheticFrameStream`` uses internally — box
    size from the frame dimensions, horizontal position from linear
    progress across the stream, vertical centering — from public metadata
    only. It never inspects the frame pixels, so a detector that recovers
    this box is genuinely processing the rendered data.
    """
    total_frames = stream.metadata.total_frames
    assert total_frames is not None

    box_size = max(min(stream.metadata.width, stream.metadata.height) // 8, 1)
    max_x = stream.metadata.width - box_size
    if total_frames > 1:
        progress = frame_id / (total_frames - 1)
    else:
        progress = 0.0
    x = round(progress * max_x)
    y = (stream.metadata.height - box_size) // 2
    return BoundingBox(
        x_min=float(x),
        y_min=float(y),
        x_max=float(x + box_size),
        y_max=float(y + box_size),
    )


def ground_truth_for(frame, box: BoundingBox) -> GroundTruthAnnotation:
    return GroundTruthAnnotation(
        frame=FrameRef(
            source_id="synthetic-stream",
            frame_index=frame.frame_id,
            timestamp=frame.timestamp_ms,
        ),
        boxes=[box],
        labels=[GROUND_TRUTH_LABEL],
        track_ids=[0],
    )


class TestStreamDetectEvaluate:
    def test_precision_and_recall_are_perfect_across_all_frames(self) -> None:
        stream = SyntheticFrameStream(width=64, height=48, fps=10.0, num_frames=6)
        detector = SyntheticBoxDetector()
        for frame in stream:
            expected = expected_object_box(stream, frame.frame_id)
            detections = detector.detect(frame)
            assert len(detections) == 1
            assert detections[0].bounding_box == expected
            precision, recall = precision_recall_at_iou(
                detections, ground_truth_for(frame, expected)
            )
            assert precision == pytest.approx(1.0)
            assert recall == pytest.approx(1.0)

    def test_empty_object_stream_reports_zero_against_zero(self) -> None:
        # No object rendered: the detector emits nothing and ground truth is
        # empty, so PR-002's documented both-empty convention yields
        # (0.0, 0.0) rather than erroring.
        stream = SyntheticFrameStream(
            width=32, height=32, num_frames=4, add_moving_object=False
        )
        detector = SyntheticBoxDetector()
        for frame in stream:
            detections = detector.detect(frame)
            assert detections == []
            ground_truth = GroundTruthAnnotation(
                frame=FrameRef(
                    source_id="synthetic-empty",
                    frame_index=frame.frame_id,
                    timestamp=frame.timestamp_ms,
                ),
                boxes=[],
                labels=[],
                track_ids=[],
            )
            assert precision_recall_at_iou(detections, ground_truth) == (0.0, 0.0)
