"""Frame provider abstraction and deterministic synthetic stream (PR-003, PR-007).

``BaseFrameProvider`` is the ingestion-side contract every video source
fulfills: expose ``StreamMetadata``, yield ``FrameData`` one frame at a time
via ``read_next``, support the iterator protocol, and guarantee resource
cleanup through the context manager protocol.

``SyntheticFrameStream`` and ``MultiObjectSyntheticFrameStream`` are deterministic,
dependency-light implementations that generate synthetic frames in pure NumPy —
no video file, no capture device. They exist so the whole pipeline is testable in
CI without media fixtures, and so every consumer of frames has a reproducible stream
to run against (see docs/adr/0003-video-ingestion-and-streaming.md and
docs/adr/0007-spatial-reidentification.md).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Container, Iterator
from dataclasses import dataclass
from types import TracebackType

import numpy as np

from sentinel_vision.ingestion.contracts import FrameData, ImageArray, StreamMetadata


@dataclass(frozen=True)
class SyntheticObjectConfig:
    """Configuration for an independent object in a synthetic frame stream (PR-007).

    Attributes:
        start_x: Initial horizontal pixel coordinate.
        start_y: Initial vertical pixel coordinate.
        velocity_x: Per-frame horizontal velocity in pixels.
        velocity_y: Per-frame vertical velocity in pixels.
        width: Box width in pixels (0 for default auto-scaled size).
        height: Box height in pixels (0 for default auto-scaled size).
        active_frames: Optional container of frame IDs during which the object is visible.
            If None, the object is visible on all frames.
    """

    start_x: float = 0.0
    start_y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    width: int = 0
    height: int = 0
    active_frames: Container[int] | None = None


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
    When ``add_moving_object`` is True, one or more filled white boxes travel across
    the frame; their positions are a pure function of ``frame_id``, so identical
    construction parameters always produce byte-identical frames (ADR-0003, ADR-0007).

    Single-object trajectory formula (when ``num_objects == 1`` and ``objects`` is None):
    - ``box_size = max(min(width, height) // 8, 1)``
    - ``max_x = width - box_size``
    - ``progress = frame_id / (num_frames - 1)`` if ``num_frames > 1`` else ``0.0``
    - ``x = round(progress * max_x)``
    - ``y = (height - box_size) // 2``

    Multi-object default trajectory formula (when ``num_objects > 1`` and ``objects`` is None):
    For object $i \\in \\{0, \\dots, \\text{num\\_objects} - 1\\}$:
    - ``box_size = max(min(width, height) // 8, 1)``
    - ``y_i = round(((i + 1) / (num_objects + 1)) * (height - box_size))``
    - ``progress = frame_id / (num_frames - 1)`` if ``num_frames > 1`` else ``0.0``
    - If $i$ is even: ``x_i = round(progress * (width - box_size))`` (left-to-right)
    - If $i$ is odd: ``x_i = round((1.0 - progress) * (width - box_size))`` (right-to-left)

    Alternatively, custom ``SyntheticObjectConfig`` instances can be supplied in ``objects``
    to define explicit starting positions, velocities, box dimensions, and active frame
    intervals (supporting disappearances, occlusions, and reappearances).

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
        num_objects: int = 1,
        objects: list[SyntheticObjectConfig] | None = None,
    ) -> None:
        if width <= 0:
            raise ValueError(f"width ({width}) must be > 0")
        if height <= 0:
            raise ValueError(f"height ({height}) must be > 0")
        if fps <= 0.0:
            raise ValueError(f"fps ({fps}) must be > 0.0")
        if num_frames <= 0:
            raise ValueError(f"num_frames ({num_frames}) must be > 0")
        if num_objects <= 0:
            raise ValueError(f"num_objects ({num_objects}) must be > 0")

        self._width = width
        self._height = height
        self._fps = fps
        self._num_frames = num_frames
        self._add_moving_object = add_moving_object
        self._num_objects = num_objects
        self._objects = objects
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

        default_box_size = max(min(self._width, self._height) // 8, 1)

        if self._objects is not None:
            for obj in self._objects:
                if obj.active_frames is not None and frame_id not in obj.active_frames:
                    continue
                w = obj.width if obj.width > 0 else default_box_size
                h = obj.height if obj.height > 0 else default_box_size
                x = round(obj.start_x + obj.velocity_x * frame_id)
                y = round(obj.start_y + obj.velocity_y * frame_id)

                y_start = max(0, y)
                y_end = min(self._height, y + h)
                x_start = max(0, x)
                x_end = min(self._width, x + w)
                if y_start < y_end and x_start < x_end:
                    image[y_start:y_end, x_start:x_end] = 255
            return image

        if self._num_objects == 1:
            max_x = self._width - default_box_size
            progress = frame_id / (self._num_frames - 1) if self._num_frames > 1 else 0.0
            x = round(progress * max_x)
            y = (self._height - default_box_size) // 2
            image[y : y + default_box_size, x : x + default_box_size] = 255
            return image

        progress = frame_id / (self._num_frames - 1) if self._num_frames > 1 else 0.0
        max_x = self._width - default_box_size
        for i in range(self._num_objects):
            y_i = round(((i + 1) / (self._num_objects + 1)) * (self._height - default_box_size))
            if i % 2 == 0:
                x_i = round(progress * max_x)
            else:
                x_i = round((1.0 - progress) * max_x)

            y_start = max(0, y_i)
            y_end = min(self._height, y_i + default_box_size)
            x_start = max(0, x_i)
            x_end = min(self._width, x_i + default_box_size)
            if y_start < y_end and x_start < x_end:
                image[y_start:y_end, x_start:x_end] = 255

        return image


class MultiObjectSyntheticFrameStream(SyntheticFrameStream):
    """Deterministic, in-memory multi-object frame stream for testing (PR-007).

    Supports ``num_objects >= 1``, each with an independent linear trajectory
    so multiple boxes can be present, moving independently in the same frame.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        num_frames: int = 100,
        num_objects: int = 2,
        add_moving_object: bool = True,
        objects: list[SyntheticObjectConfig] | None = None,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            fps=fps,
            num_frames=num_frames,
            add_moving_object=add_moving_object,
            num_objects=num_objects,
            objects=objects,
        )

