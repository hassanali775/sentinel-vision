"""Detector contract for the deterministic pipeline (PR-004).

``BaseDetector`` is the detection-side boundary every object detector
fulfills: consume one ``FrameData`` and return the ``Detection`` objects
found in it. It is deliberately a single-method interface — detection is a
pure function of one frame, not a stateful resource — mirroring the
minimalism of ``BaseFrameProvider`` (docs/adr/0003-video-ingestion-and-streaming.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_vision.data.contracts import Detection
from sentinel_vision.ingestion.contracts import FrameData


class BaseDetector(ABC):
    """Common protocol for any component that detects objects in a frame.

    Subclasses implement :meth:`detect`, which consumes a single
    ``FrameData`` and returns the list of ``Detection`` objects found in
    it. Detection is a pure function of one frame: no stream state is held
    and no resources are owned, so — unlike ``BaseFrameProvider`` — no
    iterator or context-manager machinery is needed here.
    """

    @abstractmethod
    def detect(self, frame: FrameData) -> list[Detection]:
        """Return the detections found in ``frame``.

        Implementations must treat ``frame`` as read-only and return a
        fresh list of ``Detection`` objects. An empty list means no objects
        were found — never an error.
        """
