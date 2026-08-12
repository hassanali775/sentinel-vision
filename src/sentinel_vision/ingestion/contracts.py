"""Frozen frame and stream contracts for the ingestion layer (PR-003).

``FrameData`` is the unit of video acquisition: one decoded image plus its
frame counter and timestamp. ``StreamMetadata`` describes the stream those
frames come from. Both are frozen so that a frame's facts cannot be mutated
after the fact, matching the discipline established in
``sentinel_vision.data.contracts`` (PR-002).

Validation happens in ``__post_init__`` where a malformed value is a
data-quality bug rather than a value to silently accept: a negative frame
counter or timestamp, or an image that is not a 3D uint8 array with
positive dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

ImageArray = np.ndarray[Any, np.dtype[np.uint8]]


@dataclass(frozen=True, eq=False)
class FrameData:
    """A single decoded frame of a video stream.

    ``frame_id`` is the 0-indexed sequential frame counter within the
    stream. ``timestamp_ms`` is the frame's time relative to stream start,
    in milliseconds (see ADR-0003 for the exact guarantee).
    ``image`` holds the raw pixel data as a (height, width, channels) uint8
    array. ``metadata`` carries optional runtime attributes (e.g. capture
    device id) and defaults to an empty mapping.

    Immutability is enforced at runtime, not just by ``frozen=True``:
    ``image`` is defensively copied on construction and then made
    read-only, and ``metadata`` is wrapped in a read-only
    ``MappingProxyType`` in ``__post_init__``, so neither field's contents
    can be mutated in place after construction — including through a buffer
    the caller still holds. The copy is a one-time memory/CPU cost per
    frame, acceptable for determinism; it can be profiled and removed in a
    future optimization PR if the hot path ever demands it.
    """

    frame_id: int
    timestamp_ms: float
    image: ImageArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(f"frame_id ({self.frame_id}) must be non-negative")
        if self.timestamp_ms < 0.0:
            raise ValueError(
                f"timestamp_ms ({self.timestamp_ms}) must be non-negative"
            )
        if self.image.ndim != 3:
            raise ValueError(
                f"image must be 3-dimensional, got ndim={self.image.ndim}"
            )
        if self.image.dtype != np.uint8:
            raise ValueError(
                f"image must have dtype uint8, got dtype={self.image.dtype}"
            )
        height, width, channels = self.image.shape
        if height <= 0 or width <= 0 or channels <= 0:
            raise ValueError(
                f"image dimensions must be positive, got shape={self.image.shape}"
            )

        object.__setattr__(self, "image", np.array(self.image, copy=True))
        self.image.flags.writeable = False
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameData):
            return NotImplemented
        return (
            self.frame_id == other.frame_id
            and self.timestamp_ms == other.timestamp_ms
            and np.array_equal(self.image, other.image)
            and self.metadata == other.metadata
        )

    def __hash__(self) -> int:
        # Equality is array-aware (np.array_equal), so the dataclass cannot
        # auto-generate __hash__ (eq=False + custom __eq__ makes instances
        # unhashable by default). The numpy array is deliberately excluded:
        # hash relies only on the scalar identity fields, which is consistent
        # with __eq__ — equal frames always share (frame_id, timestamp_ms).
        return hash((self.frame_id, self.timestamp_ms))


@dataclass(frozen=True)
class StreamMetadata:
    """Static description of a frame stream.

    ``total_frames`` is optional because a live or unbounded source may not
    know its length up front. ``duration_sec`` is the optional computed
    stream duration in seconds (e.g. ``total_frames / fps`` for a finite
    source); when provided it must be positive.
    """

    fps: float
    width: int
    height: int
    total_frames: int | None = None
    duration_sec: float | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0.0:
            raise ValueError(f"fps ({self.fps}) must be > 0.0")
        if self.width <= 0:
            raise ValueError(f"width ({self.width}) must be > 0")
        if self.height <= 0:
            raise ValueError(f"height ({self.height}) must be > 0")
        if self.total_frames is not None and self.total_frames <= 0:
            raise ValueError(
                f"total_frames ({self.total_frames}) must be > 0 when provided"
            )
        if self.duration_sec is not None and self.duration_sec <= 0.0:
            raise ValueError(
                f"duration_sec ({self.duration_sec}) must be > 0.0 when provided"
            )
