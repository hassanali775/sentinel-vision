"""Tests for frame/stream contracts and the synthetic stream (PR-003).

Covers ``FrameData`` and ``StreamMetadata`` validation, the abstract
``BaseFrameProvider`` protocol, and every guarantee of
``SyntheticFrameStream``: exact frame count, exact zero-based timestamp
progression, consistent array shape/dtype, deterministic moving-object
rendering, and context-manager cleanup.
"""

import numpy as np
import pytest

from sentinel_vision.ingestion.contracts import FrameData, StreamMetadata
from sentinel_vision.ingestion.stream import BaseFrameProvider, SyntheticFrameStream


def make_image(height: int = 4, width: int = 6) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_frame(
    frame_id: int = 0,
    timestamp_ms: float = 0.0,
    image: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> FrameData:
    return FrameData(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        image=make_image() if image is None else image,
        metadata={} if metadata is None else metadata,
    )


class TestFrameData:
    def test_valid_frame_constructs(self) -> None:
        frame = make_frame(frame_id=3, timestamp_ms=100.0)
        assert frame.frame_id == 3
        assert frame.timestamp_ms == 100.0
        assert frame.image.shape == (4, 6, 3)
        assert frame.image.dtype == np.uint8
        assert frame.metadata == {}

    def test_metadata_defaults_to_empty_dict(self) -> None:
        assert make_frame().metadata == {}

    def test_custom_metadata_is_preserved(self) -> None:
        metadata = {"device": "cam-01", "quality": 1}
        assert make_frame(metadata=metadata).metadata == metadata

    def test_rejects_negative_frame_id(self) -> None:
        with pytest.raises(ValueError, match="frame_id"):
            make_frame(frame_id=-1)

    def test_rejects_negative_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp_ms"):
            make_frame(timestamp_ms=-0.5)

    def test_rejects_2d_image(self) -> None:
        image = np.zeros((4, 6), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-dimensional"):
            make_frame(image=image)

    def test_rejects_4d_image(self) -> None:
        image = np.zeros((1, 4, 6, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-dimensional"):
            make_frame(image=image)

    def test_rejects_non_uint8_image(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            make_frame(image=image)

    def test_rejects_zero_height_image(self) -> None:
        image = np.zeros((0, 6, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="positive"):
            make_frame(image=image)

    def test_rejects_zero_width_image(self) -> None:
        image = np.zeros((4, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="positive"):
            make_frame(image=image)

    def test_rejects_zero_channels_image(self) -> None:
        image = np.zeros((4, 6, 0), dtype=np.uint8)
        with pytest.raises(ValueError, match="positive"):
            make_frame(image=image)

    def test_equal_frames_are_equal(self) -> None:
        assert make_frame() == make_frame()

    def test_unequal_frame_ids_are_not_equal(self) -> None:
        assert make_frame(frame_id=0) != make_frame(frame_id=1)

    def test_unequal_timestamps_are_not_equal(self) -> None:
        assert make_frame(timestamp_ms=0.0) != make_frame(timestamp_ms=1.0)

    def test_unequal_images_are_not_equal(self) -> None:
        image_a = np.zeros((4, 6, 3), dtype=np.uint8)
        image_b = np.ones((4, 6, 3), dtype=np.uint8)
        assert make_frame(image=image_a) != make_frame(image=image_b)

    def test_unequal_metadata_are_not_equal(self) -> None:
        assert make_frame() != make_frame(metadata={"a": 1})

    def test_frame_is_not_equal_to_non_frame(self) -> None:
        assert make_frame() != object()

    def test_frame_data_is_hashable(self) -> None:
        frame = make_frame()
        assert hash(frame) == hash((frame.frame_id, frame.timestamp_ms))

    def test_frame_data_usable_as_set_member_and_dict_key(self) -> None:
        frame = make_frame(frame_id=1, timestamp_ms=33.333)
        frame_set = {make_frame(frame_id=1, timestamp_ms=33.333), frame}
        assert frame_set == {frame}
        lookup = {frame: "value"}
        assert lookup[make_frame(frame_id=1, timestamp_ms=33.333)] == "value"

    def test_equal_frames_have_equal_hashes(self) -> None:
        a = make_frame(frame_id=2, timestamp_ms=66.666)
        b = make_frame(frame_id=2, timestamp_ms=66.666)
        assert a == b
        assert hash(a) == hash(b)

    def test_image_is_not_writeable(self) -> None:
        assert not make_frame().image.flags.writeable

    def test_image_mutation_is_rejected(self) -> None:
        frame = make_frame()
        with pytest.raises(ValueError):
            frame.image[:] = 0

    def test_metadata_mutation_is_rejected(self) -> None:
        frame = make_frame()
        with pytest.raises(TypeError):
            frame.metadata["x"] = 1  # type: ignore[index]

    def test_metadata_is_a_read_only_mapping(self) -> None:
        from types import MappingProxyType

        assert isinstance(make_frame().metadata, MappingProxyType)

    def test_external_dict_cannot_mutate_frame(self) -> None:
        metadata = {"a": 1}
        frame = make_frame(metadata=metadata)
        metadata["b"] = 2
        assert frame.metadata == {"a": 1}

    def test_external_array_mutation_cannot_affect_frame(self) -> None:
        image = make_image()
        original = image.copy()
        frame = make_frame(image=image)
        image[:] = 255
        assert np.array_equal(frame.image, original)


class TestStreamMetadata:
    def test_valid_metadata_constructs(self) -> None:
        metadata = StreamMetadata(fps=30.0, width=640, height=480)
        assert metadata.fps == 30.0
        assert metadata.width == 640
        assert metadata.height == 480
        assert metadata.total_frames is None
        assert metadata.duration_sec is None

    def test_optional_fields_are_accepted(self) -> None:
        metadata = StreamMetadata(
            fps=30.0, width=640, height=480, total_frames=100, duration_sec=100 / 30
        )
        assert metadata.total_frames == 100
        assert metadata.duration_sec == pytest.approx(100 / 30)

    def test_rejects_zero_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            StreamMetadata(fps=0.0, width=640, height=480)

    def test_rejects_negative_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            StreamMetadata(fps=-5.0, width=640, height=480)

    def test_rejects_zero_width(self) -> None:
        with pytest.raises(ValueError, match="width"):
            StreamMetadata(fps=30.0, width=0, height=480)

    def test_rejects_negative_width(self) -> None:
        with pytest.raises(ValueError, match="width"):
            StreamMetadata(fps=30.0, width=-1, height=480)

    def test_rejects_zero_height(self) -> None:
        with pytest.raises(ValueError, match="height"):
            StreamMetadata(fps=30.0, width=640, height=0)

    def test_rejects_negative_height(self) -> None:
        with pytest.raises(ValueError, match="height"):
            StreamMetadata(fps=30.0, width=640, height=-480)

    def test_rejects_zero_total_frames(self) -> None:
        with pytest.raises(ValueError, match="total_frames"):
            StreamMetadata(fps=30.0, width=640, height=480, total_frames=0)

    def test_rejects_negative_total_frames(self) -> None:
        with pytest.raises(ValueError, match="total_frames"):
            StreamMetadata(fps=30.0, width=640, height=480, total_frames=-1)

    def test_rejects_zero_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_sec"):
            StreamMetadata(fps=30.0, width=640, height=480, duration_sec=0.0)

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_sec"):
            StreamMetadata(fps=30.0, width=640, height=480, duration_sec=-1.0)


class TestBaseFrameProvider:
    def test_abstract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseFrameProvider()  # type: ignore[abstract]


class TestSyntheticFrameStream:
    def test_yields_exactly_num_frames(self) -> None:
        stream = SyntheticFrameStream(num_frames=7)
        frames = list(stream)
        assert len(frames) == 7

    def test_default_stream_yields_100_frames(self) -> None:
        assert len(list(SyntheticFrameStream())) == 100

    def test_frame_ids_are_sequential_from_zero(self) -> None:
        stream = SyntheticFrameStream(num_frames=5)
        assert [frame.frame_id for frame in stream] == [0, 1, 2, 3, 4]

    def test_timestamps_match_frame_id_times_interval(self) -> None:
        fps = 30.0
        interval = 1000.0 / fps
        stream = SyntheticFrameStream(fps=fps, num_frames=50)
        for frame in stream:
            assert frame.timestamp_ms == pytest.approx(frame.frame_id * interval)

    def test_timestamp_increment_is_exact(self) -> None:
        fps = 30.0
        interval = 1000.0 / fps
        stream = SyntheticFrameStream(fps=fps, num_frames=50)
        timestamps = [frame.timestamp_ms for frame in stream]
        increments = {
            round(high - low, 10) for low, high in zip(timestamps, timestamps[1:], strict=False)
        }
        assert increments == {round(interval, 10)}

    def test_all_frames_have_consistent_shape(self) -> None:
        stream = SyntheticFrameStream(width=320, height=240, num_frames=10)
        for frame in stream:
            assert frame.image.shape == (240, 320, 3)

    def test_all_frames_have_uint8_dtype(self) -> None:
        stream = SyntheticFrameStream(num_frames=10)
        for frame in stream:
            assert frame.image.dtype == np.uint8

    def test_metadata_matches_construction_params(self) -> None:
        stream = SyntheticFrameStream(width=320, height=240, fps=15.0, num_frames=8)
        metadata = stream.metadata
        assert metadata.fps == 15.0
        assert metadata.width == 320
        assert metadata.height == 240
        assert metadata.total_frames == 8
        assert metadata.duration_sec == pytest.approx(8 / 15)

    def test_metadata_is_accessible_before_reading(self) -> None:
        stream = SyntheticFrameStream()
        assert stream.metadata.total_frames == 100

    def test_iterator_returns_self(self) -> None:
        stream = SyntheticFrameStream()
        assert iter(stream) is stream

    def test_read_next_returns_none_after_exhaustion(self) -> None:
        stream = SyntheticFrameStream(num_frames=2)
        assert stream.read_next() is not None
        assert stream.read_next() is not None
        assert stream.read_next() is None

    def test_next_raises_stop_iteration_after_exhaustion(self) -> None:
        stream = SyntheticFrameStream(num_frames=1)
        next(stream)
        with pytest.raises(StopIteration):
            next(stream)

    def test_is_instance_of_base_provider(self) -> None:
        assert isinstance(SyntheticFrameStream(), BaseFrameProvider)

    def test_deterministic_with_moving_object(self) -> None:
        params = dict(width=64, height=48, fps=10.0, num_frames=5)
        stream_a = SyntheticFrameStream(**params)
        stream_b = SyntheticFrameStream(**params)
        frames_a = list(stream_a)
        frames_b = list(stream_b)
        for frame_a, frame_b in zip(frames_a, frames_b, strict=True):
            assert frame_a.frame_id == frame_b.frame_id
            assert frame_a.timestamp_ms == frame_b.timestamp_ms
            assert np.array_equal(frame_a.image, frame_b.image)

    def test_deterministic_without_moving_object(self) -> None:
        params = dict(width=64, height=48, fps=10.0, num_frames=5, add_moving_object=False)
        stream_a = SyntheticFrameStream(**params)
        stream_b = SyntheticFrameStream(**params)
        for frame_a, frame_b in zip(stream_a, stream_b, strict=True):
            assert np.array_equal(frame_a.image, frame_b.image)

    def test_frames_are_black_when_object_disabled(self) -> None:
        stream = SyntheticFrameStream(num_frames=5, add_moving_object=False)
        for frame in stream:
            assert np.all(frame.image == 0)

    def test_moving_object_is_present_by_default(self) -> None:
        stream = SyntheticFrameStream(width=64, height=48, num_frames=5)
        for frame in stream:
            assert np.any(frame.image > 0)

    def test_moving_object_travels_left_to_right(self) -> None:
        stream = SyntheticFrameStream(width=64, height=48, num_frames=9)
        frames = list(stream)
        first_cols = np.argwhere(frames[0].image > 0)[:, 1]
        last_cols = np.argwhere(frames[-1].image > 0)[:, 1]
        assert first_cols.max() < last_cols.min()

    def test_moving_object_position_depends_on_frame_id(self) -> None:
        stream = SyntheticFrameStream(width=64, height=48, num_frames=9)
        frames = list(stream)
        for a, b in zip(frames, frames[1:], strict=False):
            cols_a = np.argwhere(a.image > 0)[:, 1]
            cols_b = np.argwhere(b.image > 0)[:, 1]
            assert cols_a.min() < cols_b.min()

    def test_context_manager_returns_self(self) -> None:
        stream = SyntheticFrameStream()
        with stream as entered:
            assert entered is stream

    def test_context_manager_cleanup_closes_stream(self) -> None:
        stream = SyntheticFrameStream()
        with stream:
            assert stream.read_next() is not None
        with pytest.raises(RuntimeError, match="closed"):
            stream.read_next()

    def test_context_manager_cleanup_on_exception(self) -> None:
        stream = SyntheticFrameStream()
        with pytest.raises(RuntimeError, match="boom"):
            with stream:
                raise RuntimeError("boom")
        with pytest.raises(RuntimeError, match="closed"):
            stream.read_next()

    def test_close_is_idempotent(self) -> None:
        stream = SyntheticFrameStream()
        stream.close()
        stream.close()

    def test_rejects_zero_width(self) -> None:
        with pytest.raises(ValueError, match="width"):
            SyntheticFrameStream(width=0)

    def test_rejects_negative_height(self) -> None:
        with pytest.raises(ValueError, match="height"):
            SyntheticFrameStream(height=-1)

    def test_rejects_zero_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            SyntheticFrameStream(fps=0.0)

    def test_rejects_zero_num_frames(self) -> None:
        with pytest.raises(ValueError, match="num_frames"):
            SyntheticFrameStream(num_frames=0)

    def test_single_frame_stream_produces_one_valid_frame(self) -> None:
        stream = SyntheticFrameStream(width=32, height=32, num_frames=1)
        frames = list(stream)
        assert len(frames) == 1
        assert frames[0].frame_id == 0
        assert frames[0].timestamp_ms == 0.0
        assert frames[0].image.shape == (32, 32, 3)
