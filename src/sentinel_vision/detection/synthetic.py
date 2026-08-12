"""Synthetic threshold detector for the deterministic pipeline (PR-004).

``SyntheticBoxDetector`` finds the moving object rendered by
``SyntheticFrameStream`` by thresholding pixels directly — any channel
value above zero is foreground. It does not replay the stream's known
trajectory: it genuinely processes pixel data, so precision/recall
evaluation against independently-computed ground truth actually measures
the detector rather than a mock (see
docs/adr/0004-detection-abstraction.md).
"""

from __future__ import annotations

import numpy as np

from sentinel_vision.data.contracts import BoundingBox, Detection
from sentinel_vision.detection.base import BaseDetector
from sentinel_vision.ingestion.contracts import FrameData

DEFAULT_CONFIDENCE = 1.0
SYNTHETIC_TARGET_LABEL = "synthetic_target"


class SyntheticBoxDetector(BaseDetector):
    """Detect the synthetic stream's moving object as one bounding box.

    Foreground is every pixel with any channel ``> 0``: the synthetic
    stream renders its object as a filled white box on a black background,
    so thresholding at zero exactly separates object from scene. All
    foreground pixels are enclosed in a single axis-aligned box using the
    exclusive-max convention from PR-002's contracts.

    Known simplifications:

    - ``confidence`` is always ``1.0``: no probabilistic model exists at
      this stage, so the one signal the detector has (foreground pixels
      present) yields full certainty.
    - ``class_label`` is always ``"synthetic_target"``: no class
      discrimination exists at this stage, so a single constant label is
      used.
    - Exactly one box is returned even if the foreground pixels were
      disjoint: multi-instance detection (connected components / NMS) is
      deferred, matching the synthetic stream's single-object design.
    """

    def detect(self, frame: FrameData) -> list[Detection]:
        """Return the enclosing box of all foreground pixels in ``frame``.

        An empty frame (no pixel above zero) yields an empty list — zero
        detections, never an error. ``frame.image`` is only read, never
        mutated.
        """
        foreground = np.argwhere(frame.image > 0)
        if foreground.size == 0:
            return []

        rows = foreground[:, 0]
        cols = foreground[:, 1]
        bounding_box = BoundingBox(
            x_min=float(cols.min()),
            y_min=float(rows.min()),
            x_max=float(cols.max() + 1),
            y_max=float(rows.max() + 1),
        )
        return [
            Detection(
                bounding_box=bounding_box,
                confidence=DEFAULT_CONFIDENCE,
                class_label=SYNTHETIC_TARGET_LABEL,
            )
        ]
