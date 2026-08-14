"""Synthetic threshold detector for the deterministic pipeline (PR-004, PR-007).

``SyntheticBoxDetector`` finds moving objects rendered by ``SyntheticFrameStream``
and ``MultiObjectSyntheticFrameStream`` by thresholding pixels directly — any channel
value above zero is foreground. It does not replay the stream's known trajectory:
it genuinely processes pixel data, so precision/recall evaluation against
independently-computed ground truth actually measures the detector rather than a mock
(see docs/adr/0004-detection-abstraction.md).
"""

from __future__ import annotations

import numpy as np

from sentinel_vision.data.contracts import BoundingBox, Detection
from sentinel_vision.detection.base import BaseDetector
from sentinel_vision.ingestion.contracts import FrameData, ImageArray

DEFAULT_CONFIDENCE = 1.0
SYNTHETIC_TARGET_LABEL = "synthetic_target"


def _extract_boxes(image: ImageArray) -> list[BoundingBox]:
    mask = np.asarray((image > 0).any(axis=2), dtype=bool)
    if not np.any(mask):
        return []

    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    boxes: list[BoundingBox] = []

    for r in range(height):
        for c in range(width):
            if mask[r, c].item() and not visited[r, c].item():
                min_r, max_r = r, r
                min_c, max_c = c, c
                queue = [(r, c)]
                visited[r, c] = True

                while queue:
                    curr_r, curr_c = queue.pop()
                    if curr_r < min_r:
                        min_r = curr_r
                    if curr_r > max_r:
                        max_r = curr_r
                    if curr_c < min_c:
                        min_c = curr_c
                    if curr_c > max_c:
                        max_c = curr_c

                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                if mask[nr, nc].item() and not visited[nr, nc].item():
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))

                boxes.append(
                    BoundingBox(
                        x_min=float(min_c),
                        y_min=float(min_r),
                        x_max=float(max_c + 1),
                        y_max=float(max_r + 1),
                    )
                )

    boxes.sort(key=lambda b: (b.y_min, b.x_min))
    return boxes


class SyntheticBoxDetector(BaseDetector):
    """Detect the synthetic stream's moving object(s) as bounding boxes.

    Foreground is every pixel with any channel ``> 0``: the synthetic
    stream renders its object as a filled white box on a black background,
    so thresholding at zero exactly separates object from scene. All
    foreground pixels are enclosed in axis-aligned boxes using the
    exclusive-max convention from PR-002's contracts.

    When ``multi_instance`` is False (default for backward compatibility),
    all foreground pixels are enclosed in a single bounding box.
    When ``multi_instance`` is True, distinct connected components of foreground
    pixels are extracted as separate bounding boxes (PR-007).
    """

    def __init__(self, multi_instance: bool = False) -> None:
        self._multi_instance = multi_instance

    def detect(self, frame: FrameData) -> list[Detection]:
        """Return enclosing box(es) of foreground pixels in ``frame``.

        An empty frame (no pixel above zero) yields an empty list — zero
        detections, never an error. ``frame.image`` is only read, never
        mutated.
        """
        foreground = np.argwhere(frame.image > 0)
        if foreground.size == 0:
            return []

        if not self._multi_instance:
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

        boxes = _extract_boxes(frame.image)
        return [
            Detection(
                bounding_box=box,
                confidence=DEFAULT_CONFIDENCE,
                class_label=SYNTHETIC_TARGET_LABEL,
            )
            for box in boxes
        ]


