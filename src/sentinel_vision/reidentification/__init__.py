"""Spatial re-identification and candidate retention package (PR-007, ADR-0011).

Provides spatial/motion-only re-identification (``ReidentificationCandidate``,
``SpatialReidentifier``) to re-link newly observed detections to recently retired
entities based on finite-difference velocity prediction without appearance modeling,
plus ``plausibility_score``, the single shared plausibility rule used by both the
retired candidate pool and the active-entity pool (ADR-0011).
"""

__all__ = [
    "ReidentificationCandidate",
    "SpatialReidentifier",
    "plausibility_score",
]

from sentinel_vision.reidentification.spatial import (
    ReidentificationCandidate,
    SpatialReidentifier,
    plausibility_score,
)
