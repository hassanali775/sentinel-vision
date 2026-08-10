"""Frame ingestion contracts and stream providers for Sentinel Vision (PR-003).

This package defines the acquisition boundary of the deterministic
pipeline: the immutable ``FrameData``/``StreamMetadata`` contracts, the
``BaseFrameProvider`` protocol every video source implements, and
``SyntheticFrameStream`` — a deterministic, NumPy-only stream used for
testing and CI. See docs/adr/0003-video-ingestion-and-streaming.md.
"""

__all__ = [
    "BaseFrameProvider",
    "FrameData",
    "ImageArray",
    "StreamMetadata",
    "SyntheticFrameStream",
]

from sentinel_vision.ingestion.contracts import FrameData, ImageArray, StreamMetadata
from sentinel_vision.ingestion.stream import BaseFrameProvider, SyntheticFrameStream
