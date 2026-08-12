"""Detection abstraction and synthetic threshold detector (PR-004).

This package defines the detection boundary of the deterministic
pipeline: the ``BaseDetector`` protocol every detector implements, and
``SyntheticBoxDetector`` — a NumPy-only threshold detector that finds the
synthetic stream's moving object by its pixels. See
docs/adr/0004-detection-abstraction.md.
"""

__all__ = [
    "BaseDetector",
    "SyntheticBoxDetector",
]

from sentinel_vision.detection.base import BaseDetector
from sentinel_vision.detection.synthetic import SyntheticBoxDetector
