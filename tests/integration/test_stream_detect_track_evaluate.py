"""End-to-end stream -> detect -> track -> evaluate pipeline (PR-005).

Runs every frame of a ``SyntheticFrameStream`` through
``SyntheticBoxDetector`` and ``GreedyIoUTracker``, then scores the tracked
predictions with PR-005's ``calculate_mota`` / ``calculate_idf1`` against
ground truth recomputed independently in this test (re-deriving the
stream's rendering formula from public metadata, not the frame pixels). A
second test exercises track expiration: a multi-frame gap makes the greedy
tracker drop and re-create the track.
"""

import pytest

from sentinel_vision.data.contracts import (
    BoundingBox,
    FrameRef,
    GroundTruthAnnotation,
    TrackedDetection,
)
from sentinel_vision.detection.synthetic import SyntheticBoxDetector
from sentinel_vision.evaluation.tracking_metrics import calculate_idf1, calculate_mota
from sentinel_vision.ingestion.stream import SyntheticFrameStream
from sentinel_vision.tracking.greedy import GreedyIoUTracker

GROUND_TRUTH_LABEL = "synthetic_target"


def expected_object_box(stream: SyntheticFrameStream, frame_id: int) -> BoundingBox:
    """Independently recompute the moving-object box for ``frame_id``.

    Re-derives the formula ``SyntheticFrameStream`` uses internally — box
    size from the frame dimensions, horizontal position from linear
    progress across the stream, vertical centering — from public metadata
    only. It never inspects the frame pixels, so a tracker that follows the
    detector's pixel-derived boxes is genuinely processing the rendered
    data.
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


class TestStreamDetectTrackEvaluate:
    def test_perfect_pipeline_scores_one_on_mota_and_idf1(self) -> None:
        # Parameters chosen so consecutive object positions overlap well
        # above the 0.5 IoU threshold: the tracker keeps a single track id.
        stream = SyntheticFrameStream(width=128, height=96, fps=10.0, num_frames=40)
        detector = SyntheticBoxDetector()
        tracker = GreedyIoUTracker(iou_threshold=0.5, max_age=1)
        ground_truth: list[GroundTruthAnnotation] = []
        predictions: list[list[TrackedDetection]] = []
        for frame in stream:
            expected = expected_object_box(stream, frame.frame_id)
            tracked = tracker.track(frame, detector.detect(frame))
            assert len(tracked) == 1
            assert tracked[0].track_id == 0
            ground_truth.append(
                GroundTruthAnnotation(
                    frame=FrameRef(
                        source_id="synthetic-stream",
                        frame_index=frame.frame_id,
                        timestamp=frame.timestamp_ms,
                    ),
                    boxes=[expected],
                    labels=[GROUND_TRUTH_LABEL],
                    track_ids=[0],
                )
            )
            predictions.append(tracked)

        assert calculate_mota(ground_truth, predictions) == pytest.approx(1.0)
        assert calculate_idf1(ground_truth, predictions) == pytest.approx(1.0)

    def test_track_is_recreated_after_gap(self) -> None:
        # Two consecutive frames without detections exceed the tracker's
        # max_age=1, so the track is dropped; when the object reappears it
        # starts a new track id and that id stays stable afterwards.
        stream = SyntheticFrameStream(width=128, height=96, fps=10.0, num_frames=40)
        detector = SyntheticBoxDetector()
        tracker = GreedyIoUTracker(iou_threshold=0.5, max_age=1)
        frames = [next(stream) for _ in range(5)]
        results: list[list[TrackedDetection]] = []
        for index, frame in enumerate(frames):
            detections = [] if index in (1, 2) else detector.detect(frame)
            results.append(tracker.track(frame, detections))

        assert [len(r) for r in results] == [1, 0, 0, 1, 1]
        assert results[0][0].track_id == 0
        assert results[3][0].track_id == 1
        assert results[4][0].track_id == 1
