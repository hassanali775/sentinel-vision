"""Real video file ingestion via OpenCV (PR-010).

``VideoFileFrameProvider`` fulfills ``BaseFrameProvider`` for a real video
file by wrapping ``cv2.VideoCapture``. It is the first ingestion source that
is not synthetic, and — together with ``YoloDetector`` and the visualization
layer — the first consumer of the OpenCV dependency boundary opened by this
PR (see docs/adr/0010-real-video-and-detector-integration.md).

Color-order contract (load-bearing):
    OpenCV's ``VideoCapture.read`` returns frames in native BGR channel
    order. This provider converts every frame BGR -> RGB before wrapping it
    in ``FrameData``, because the pipeline standardizes on RGB:
    ``SyntheticFrameStream`` renders ``(height, width, 3)`` uint8 arrays
    with the conventional RGB channel meaning, and the visualization layer
    draws on that RGB layout (its color triples are RGB, see
    ``sentinel_vision.visualization.render``). Keeping exactly one color
    order end to end is what makes a rendered box's color correct in the
    final output video; the CLI pipeline converts RGB -> BGR only at the
    ``cv2.VideoWriter`` boundary.

Timestamp contract:
    ``timestamp_ms`` is computed as ``frame_id * (1000.0 / fps)`` — the exact
    ``SyntheticFrameStream``/ADR-0003 convention — rather than trusting
    ``CAP_PROP_POS_MSEC``, which is codec-dependent and can be imprecise.
    This keeps timestamps deterministic and consistent across synthetic and
    real sources.
"""

from __future__ import annotations

import math
import os

import numpy as np
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    VideoCapture,
    cvtColor,
)

from sentinel_vision.ingestion.contracts import FrameData, StreamMetadata
from sentinel_vision.ingestion.stream import BaseFrameProvider


class VideoFileFrameProvider(BaseFrameProvider):
    """Read ``FrameData`` frames from a real video file via OpenCV.

    Opens the file eagerly at construction and raises a clear error if it
    cannot be opened — OpenCV's own failure modes for a missing or
    undecodable file are vague and are deliberately not allowed to leak
    through. ``metadata`` reflects the file's fps/width/height/frame count
    read from the capture properties.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = str(path)
        cap = VideoCapture(self._path)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(
                f"could not open video file '{self._path}': OpenCV failed to "
                "open it (the file may be missing, unreadable, or encoded "
                "with an unsupported codec). Verify the path and that the "
                "file is a decodable video."
            )
        self._cap = cap
        self._closed = False
        self._exhausted = False
        self._frame_id = 0

        fps = float(cap.get(CAP_PROP_FPS))
        if math.isnan(fps) or fps <= 0.0:
            cap.release()
            raise ValueError(
                f"could not determine a usable fps for video '{self._path}' "
                f"(OpenCV reported fps={fps!r}). A positive framerate is "
                "required to compute frame timestamps."
            )
        self._fps = fps

        width = int(cap.get(CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(CAP_PROP_FRAME_HEIGHT))

        frame_count_raw = cap.get(CAP_PROP_FRAME_COUNT)
        if math.isfinite(frame_count_raw) and frame_count_raw > 0:
            total_frames = int(frame_count_raw)
            duration_sec = total_frames / fps
        else:
            total_frames = None
            duration_sec = None

        try:
            self._metadata = StreamMetadata(
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                duration_sec=duration_sec,
            )
        except ValueError:
            cap.release()
            raise

    @property
    def metadata(self) -> StreamMetadata:
        """Static description of the video file, read at construction."""
        return self._metadata

    def read_next(self) -> FrameData | None:
        """Return the next frame, or ``None`` once the stream is exhausted.

        Each frame's ``image`` is the file's BGR frame converted to RGB (see
        the module docstring for why), with ``frame_id`` sequential from zero
        and ``timestamp_ms = frame_id * (1000.0 / fps)``.
        """
        if self._closed:
            raise RuntimeError("cannot read from a closed stream")
        if self._exhausted:
            return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._exhausted = True
            return None

        # np.asarray with an explicit uint8 dtype normalizes cv2's return
        # typing to ImageArray; the source is already uint8 so this is a
        # view/no-op, not a second copy.
        rgb_frame = np.asarray(cvtColor(frame, COLOR_BGR2RGB), dtype=np.uint8)
        frame_data = FrameData(
            frame_id=self._frame_id,
            timestamp_ms=self._frame_id * (1000.0 / self._fps),
            image=rgb_frame,
        )
        self._frame_id += 1
        return frame_data

    def close(self) -> None:
        """Release the capture. Idempotent; reads are rejected once closed."""
        if not self._closed:
            self._cap.release()
            self._closed = True
