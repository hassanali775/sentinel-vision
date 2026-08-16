"""Tests for the real-video frame provider (PR-010).

``VideoFileFrameProvider`` wraps ``cv2.VideoCapture``. These tests stub the
capture so the FrameData contract mapping (frame_id sequencing, timestamp
formula, BGR -> RGB conversion) and error handling (file-not-found) are
verified without requiring a real video file in CI.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from sentinel_vision.ingestion import video as video_module
from sentinel_vision.ingestion.stream import BaseFrameProvider
from sentinel_vision.ingestion.video import VideoFileFrameProvider


def make_bgr_frame() -> np.ndarray:
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30)  # BGR: blue=10, green=20, red=30
    frame[1, 2] = (255, 0, 0)  # BGR pure blue
    return frame


def make_props(
    fps: float = 30.0,
    width: float = 6.0,
    height: float = 4.0,
    count: float = 10.0,
) -> dict[int, float]:
    return {
        video_module.CAP_PROP_FPS: fps,
        video_module.CAP_PROP_FRAME_WIDTH: width,
        video_module.CAP_PROP_FRAME_HEIGHT: height,
        video_module.CAP_PROP_FRAME_COUNT: count,
    }


def patch_capture(props: dict[int, float], frames: list[np.ndarray], opened: bool = True):
    created: list = []

    class _FakeCapture:
        def __init__(self, path: str) -> None:
            self.path = path
            self.released = False
            self._index = 0
            created.append(self)

        def isOpened(self) -> bool:
            return opened

        def get(self, prop: int) -> float:
            value = props.get(prop)
            return 0.0 if value is None else value

        def read(self):
            if self._index >= len(frames):
                return False, None
            frame = frames[self._index]
            self._index += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    patcher = mock.patch.object(video_module, "VideoCapture", new=_FakeCapture)
    return patcher, created


class TestVideoFileFrameProvider:
    def test_raises_clear_error_when_file_cannot_be_opened(self) -> None:
        patcher, created = patch_capture({}, [], opened=False)
        with patcher:
            with pytest.raises(FileNotFoundError, match="missing-clip.mp4"):
                VideoFileFrameProvider("missing-clip.mp4")
        assert created[0].released

    def test_metadata_maps_cv2_properties(self) -> None:
        patcher, _ = patch_capture(
            make_props(fps=25.0, width=320.0, height=240.0, count=40.0), [make_bgr_frame()]
        )
        with patcher:
            provider = VideoFileFrameProvider("clip.mp4")
            assert provider.metadata.fps == 25.0
            assert provider.metadata.width == 320
            assert provider.metadata.height == 240
            assert provider.metadata.total_frames == 40
            assert provider.metadata.duration_sec == pytest.approx(40 / 25)

    def test_unknown_frame_count_yields_none_total(self) -> None:
        patcher, _ = patch_capture(make_props(count=0.0), [make_bgr_frame()])
        with patcher:
            provider = VideoFileFrameProvider("clip.mp4")
            assert provider.metadata.total_frames is None
            assert provider.metadata.duration_sec is None

    def test_unusable_fps_raises(self) -> None:
        patcher, created = patch_capture(make_props(fps=0.0), [make_bgr_frame()])
        with patcher:
            with pytest.raises(ValueError, match="fps"):
                VideoFileFrameProvider("clip.mp4")
        assert created[0].released

    def test_frame_contract_mapping(self) -> None:
        frames = [make_bgr_frame() for _ in range(3)]
        patcher, _ = patch_capture(make_props(fps=30.0), frames)
        with patcher:
            provider = VideoFileFrameProvider("clip.mp4")
            assert isinstance(provider, BaseFrameProvider)
            for expected_id in range(3):
                frame = provider.read_next()
                assert frame is not None
                assert frame.frame_id == expected_id
                assert frame.timestamp_ms == pytest.approx(expected_id * (1000.0 / 30.0))
                assert frame.image.shape == (4, 6, 3)
                assert frame.image.dtype == np.uint8
                assert not frame.image.flags.writeable
            assert provider.read_next() is None
            assert provider.read_next() is None

    def test_bgr_to_rgb_conversion(self) -> None:
        patcher, _ = patch_capture(make_props(), [make_bgr_frame()])
        with patcher:
            provider = VideoFileFrameProvider("clip.mp4")
            frame = provider.read_next()
            assert frame is not None
            assert frame.image[0, 0].tolist() == [30, 20, 10]
            assert frame.image[1, 2].tolist() == [0, 0, 255]

    def test_close_is_idempotent_and_rejects_reads(self) -> None:
        patcher, created = patch_capture(make_props(), [make_bgr_frame()])
        with patcher:
            provider = VideoFileFrameProvider("clip.mp4")
            provider.close()
            provider.close()
            assert created[0].released
            with pytest.raises(RuntimeError, match="closed"):
                provider.read_next()

    def test_context_manager_cleanup_closes_provider(self) -> None:
        patcher, created = patch_capture(make_props(), [make_bgr_frame()])
        with patcher:
            with VideoFileFrameProvider("clip.mp4") as provider:
                assert provider.read_next() is not None
        assert created[0].released
        with pytest.raises(RuntimeError, match="closed"):
            provider.read_next()
