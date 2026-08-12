"""Tracker contract for the deterministic pipeline (PR-005).

``BaseTracker`` is the tracking-side boundary every tracker fulfills:
consume the detections of one frame, associate them with the tracker's
internal tracks, and return each detection tagged with a stable
``track_id``. Unlike ``BaseDetector`` (a pure function of one frame,
ADR-0004), tracking is inherently stateful — the association at frame ``t``
depends on what was seen at frame ``t-1`` — so callers must invoke
``track`` once per frame in stream order (see
docs/adr/0005-tracker-and-evaluation-harness.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_vision.data.contracts import Detection, TrackedDetection
from sentinel_vision.ingestion.contracts import FrameData


class BaseTracker(ABC):
    """Common protocol for any component that associates detections over time.

    Subclasses implement :meth:`track`, which consumes the detections for
    one frame and returns them tagged with persistent track ids. Tracking
    holds association state between calls, so the method must be called
    once per frame, in stream order, and never concurrently.
    """

    @abstractmethod
    def track(
        self, frame: FrameData, detections: list[Detection]
    ) -> list[TrackedDetection]:
        """Associate ``detections`` in ``frame`` with tracks and tag each.

        Must be called once per frame in stream order. Returns a
        ``TrackedDetection`` for every input detection; an empty list means
        no detections were received, never an error.
        """
