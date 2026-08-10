"""Detection-level evaluation for Sentinel Vision (PR-002).

This package measures how well a detector agrees with hand-labeled ground
truth at the level of a single frame: box overlap (``geometry.iou``) and
thresholded precision/recall over a frame's predictions
(``detection_metrics``). Tracking-level and event-level metrics are
intentionally out of scope here — see docs/adr/0002-data-and-evaluation
-strategy.md for when and why they land.
"""

__all__ = ["iou", "precision_recall_at_iou"]

from sentinel_vision.evaluation.detection_metrics import precision_recall_at_iou
from sentinel_vision.evaluation.geometry import iou
