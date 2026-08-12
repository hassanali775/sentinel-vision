"""Greedy IoU tracker for the deterministic pipeline (PR-005).

``GreedyIoUTracker`` performs frame-to-frame data association by IoU: each
detection is matched to the still-available active track whose last box
overlaps it most, provided the overlap meets ``iou_threshold``. Matched
detections inherit the track's id; unmatched detections start new tracks
with monotonically increasing ids. This is deliberately the simplest
baseline — no motion model, no appearance — chosen in ADR-0005 because the
synthetic object is a fully-known deterministic trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel_vision.data.contracts import BoundingBox, Detection, TrackedDetection
from sentinel_vision.evaluation.geometry import iou
from sentinel_vision.ingestion.contracts import FrameData
from sentinel_vision.tracking.base import BaseTracker


@dataclass
class _ActiveTrack:
    track_id: int
    last_box: BoundingBox
    age: int


class GreedyIoUTracker(BaseTracker):
    """Greedy maximum-IoU data association, one track per matched detection.

    Detections are processed in descending confidence order (stable sort,
    so ties keep input order). Each detection takes the available track
    with the strictly highest IoU that meets ``iou_threshold``; a track
    matched by one detection is removed from consideration for the rest of
    the frame, so at most one detection is assigned per track per frame.
    Ties in IoU resolve to the track with the lowest ``track_id`` (creation
    order).

    ``max_age`` is the number of consecutive frames a track may go
    unmatched before it is dropped; a track that reappears after being
    dropped starts a new, never-reused track id. ``max_age=0`` drops a
    track after a single unmatched frame.
    """

    def __init__(self, iou_threshold: float = 0.5, max_age: int = 1) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                f"iou_threshold ({iou_threshold}) must be within [0.0, 1.0]"
            )
        if max_age < 0:
            raise ValueError(f"max_age ({max_age}) must be >= 0")

        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._tracks: list[_ActiveTrack] = []
        self._next_track_id = 0

    def track(
        self, frame: FrameData, detections: list[Detection]
    ) -> list[TrackedDetection]:
        """Associate ``detections`` in ``frame`` with tracks.

        Must be called once per frame in stream order; the tracker holds
        association state between calls. ``frame`` supplies the temporal
        context (frame id/timestamp) of this association step.
        """
        available = list(self._tracks)
        matched_track_ids: set[int] = set()
        results: list[TrackedDetection] = []

        for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
            best_track: _ActiveTrack | None = None
            best_iou = 0.0
            for track in available:
                candidate = iou(detection.bounding_box, track.last_box)
                if candidate < self._iou_threshold:
                    continue
                if candidate > best_iou:
                    best_iou = candidate
                    best_track = track
            if best_track is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                new_track = _ActiveTrack(
                    track_id=track_id,
                    last_box=detection.bounding_box,
                    age=0,
                )
                self._tracks.append(new_track)
                matched_track_ids.add(track_id)
                results.append(TrackedDetection(detection=detection, track_id=track_id))
            else:
                available.remove(best_track)
                best_track.last_box = detection.bounding_box
                best_track.age = 0
                matched_track_ids.add(best_track.track_id)
                results.append(
                    TrackedDetection(detection=detection, track_id=best_track.track_id)
                )

        for track in self._tracks:
            if track.track_id not in matched_track_ids:
                track.age += 1

        self._tracks = [t for t in self._tracks if t.age <= self._max_age]
        return results

