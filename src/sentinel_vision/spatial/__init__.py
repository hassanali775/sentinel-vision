"""Spatial workspace model package (PR-008).

Provides per-frame spatial facts only — zone membership and pairwise
distances between entity box centers. No temporal judgment, no thresholds,
no violations: that layer is PR-009. See
docs/adr/0008-spatial-workspace-model.md.
"""

__all__ = [
    "SpatialFrameObservation",
    "WorkspaceModel",
    "Zone",
]

from sentinel_vision.spatial.workspace import (
    SpatialFrameObservation,
    WorkspaceModel,
)
from sentinel_vision.spatial.zone import Zone
