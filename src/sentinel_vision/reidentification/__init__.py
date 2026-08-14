"""Spatial re-identification and candidate retention package (PR-007).

Provides spatial/motion-only re-identification (``ReidentificationCandidate``,
``SpatialReidentifier``) to re-link newly observed detections to recently retired
entities based on finite-difference velocity prediction without appearance modeling.
"""

__all__ = [
    "ReidentificationCandidate",
    "SpatialReidentifier",
]

from sentinel_vision.reidentification.spatial import (
    ReidentificationCandidate,
    SpatialReidentifier,
)
