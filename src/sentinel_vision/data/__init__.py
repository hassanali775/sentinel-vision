"""Data contracts for Sentinel Vision (PR-002).

The contracts module is the shared, versioned vocabulary between dataset
curation, the detection abstraction (PR-004), and the evaluation layer:
anything that describes a frame or a detection speaks in these types.
"""

__all__ = [
    "BoundingBox",
    "Detection",
    "FrameRef",
    "GroundTruthAnnotation",
]

from sentinel_vision.data.contracts import (
    BoundingBox,
    Detection,
    FrameRef,
    GroundTruthAnnotation,
)
