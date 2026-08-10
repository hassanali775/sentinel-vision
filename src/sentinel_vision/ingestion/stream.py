"""Frame provider abstraction and deterministic synthetic stream (PR-003).

``BaseFrameProvider`` is the ingestion-side contract every video source
fulfills: expose ``StreamMetadata``, yield ``FrameData`` one frame at a time
via ``read_next``, support the iterator protocol, and guarantee resource
cleanup through the context manager protocol.

``SyntheticFrameStream`` is a deterministic, dependency-light implementation
that generates synthetic frames in pure NumPy — no video file, no capture
device. It exists so the whole pipeline is testable in CI without media
fixtures, and so every consumer of frames has a reproducible stream to run
against (see docs/adr/0003-video-ingestion-and-streaming.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType

import numpy as np

from sentinel_vision.ingestion.contracts import FrameData, ImageArray, StreamMetadata


class BaseFrameProvider(ABC):
    """Common protocol for any source of ``FrameData``.

    Subclasses implement :attr:`metadata` and :meth:`read_next`. The iterator
    and context-manager protocols are provided here so that every provider is
    consumable the same way: ``for frame in provider`` and
    ``with provider as stream``. Exhaustion is signalled by ``read_next``
    returning ``None`` (never by raising).
    """

    @property
    @abstractmethod
    def metadata(self) -> StreamMetadata:
        """Static description of the stream this provider yields."""

    @abstractmethod
    def read_next(self) -> FrameData | None:
        """Return the next frame, or ``None`` once the stream is exhausted."""

    def __iter__(self) -> Iterator[FrameData]:
        return self

    def __next__(self) -> FrameData:
        frame = self.read_next()
        if frame is None:
            raise StopIteration
        return frame

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the provider. Idempotent.

        Subclasses that hold resources release them here. Providers must
        reject further reads once closed.
        """

    def __enter__(self) -> BaseFrameProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class SyntheticFrameStream(BaseFrameProvider):
    """Deterministic, in-memory frame stream for testing and development.

    Generates ``num_frames`` frames of ``(height, width, 3)`` uint8 zeros.
    When ``add_moving_object`` is True, a filled white box travels from left
    to right across the frame; its position is a pure function of
    ``frame_id``, so identical construction parameters always produce
    byte-identical frames (ADR-0003).

    ``timestamp_ms`` is exactly ``frame_id * (1000.0 / fps)``: a zero-based,
    monotonic, deterministic timestamp relative to stream start — never a
    system clock reading.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        num_frames: int = 100,
        add_moving_object: bool = True,
    ) -> None:
        if width <= 0:
            raise ValueError(f"width ({width}) must be > 0")
        if height <= 0:
            raise ValueError(f"height ({height}) must be > 0")
        if fps <= 0.0:
            raise ValueError(f"fps ({fps}) must be > 0.0")
        if num_frames <= 0:
            raise ValueError(f"num_frames ({num_frames}) must be > 0")

        self._width = width
        self._height = height
        self._fps = fps
        self._num_frames = num_frames
        self._add_moving_object = add_moving_object
        self._metadata = StreamMetadata(
            fps=fps,
            width=width,
            height=height,
            total_frames=num_frames,
            duration_sec=num_frames / fps,
        )
        self._closed = False
        self._next_frame_id = 0

    @property
    def metadata(self) -> StreamMetadata:
        return self._metadata

    def read_next(self) -> FrameData | None:
        if self._closed:
            raise RuntimeError("cannot read from a closed stream")
        if self._next_frame_id >= self._num_frames:
            return None
        frame_id = self._next_frame_id
        self._next_frame_id += 1
        return FrameData(
            frame_id=frame_id,
            timestamp_ms=frame_id * (1000.0 / self._fps),
            image=self._render_frame(frame_id),
        )

    def close(self) -> None:
        self._closed = True

    def _render_frame(self, frame_id: int) -> ImageArray:
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        if not self._add_moving_object:
            return image

        box_size = max(min(self._width, self._height) // 8, 1)
        max_x = self._width - box_size
        if self._num_frames > 1:
            progress = frame_id / (self._num_frames - 1)
        else:
            progress = 0.0
        x = round(progress * max_x)
        y = (self._height - box_size) // 2
        image[y : y + box_size, x : x + box_size] = 255
        return image
