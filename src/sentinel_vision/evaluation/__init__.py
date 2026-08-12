"""Detection and tracking evaluation for Sentinel Vision (PR-002/PR-005).

This package measures how well a detector agrees with hand-labeled ground
truth at the level of a single frame: box overlap (``geometry.iou``) and
thresholded precision/recall over a frame's predictions
(``detection_metrics``). PR-005 adds the tracking level: MOTA and IDF1 over
a frame sequence (``tracking_metrics``). Event-level metrics are
intentionally out of scope here — see docs/adr/0002-data-and-evaluation
-strategy.md and docs/adr/0005-tracker-and-evaluation-harness.md.
"""

__all__ = [
    "calculate_idf1",
    "calculate_mota",
    "iou",
    "precision_recall_at_iou",
]

from sentinel_vision.evaluation.detection_metrics import precision_recall_at_iou
from sentinel_vision.evaluation.geometry import iou
from sentinel_vision.evaluation.tracking_metrics import calculate_idf1, calculate_mota
