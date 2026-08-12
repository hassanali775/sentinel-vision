"""Tracker abstraction and greedy IoU baseline (PR-005).

This package defines the tracking boundary of the deterministic pipeline:
the ``BaseTracker`` contract every tracker implements, and
``GreedyIoUTracker`` — a frame-to-frame greedy IoU data association used as
the evaluation baseline. See docs/adr/0005-tracker-and-evaluation-harness.md.
"""

__all__ = [
    "BaseTracker",
    "GreedyIoUTracker",
]

from sentinel_vision.tracking.base import BaseTracker
from sentinel_vision.tracking.greedy import GreedyIoUTracker
