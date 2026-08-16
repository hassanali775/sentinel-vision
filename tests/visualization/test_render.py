"""Tests for the visualization render layer (PR-010).

Verifies the pure-copy contract — rendering returns a new array and the
input frame's image is provably unmodified (the same immutability-check
pattern used elsewhere in this codebase) — plus the visual elements: box
colors by state, dashed predicted boxes, zone outlines, and the OPEN-event
banner.
"""

from __future__ import annotations

import numpy as np

from sentinel_vision.data.contracts import BoundingBox
from sentinel_vision.events.event import Event, EventStatus, EventType
from sentinel_vision.ingestion.contracts import FrameData
from sentinel_vision.spatial.zone import Zone
from sentinel_vision.state.entity import EntityObservation, EntityState
from sentinel_vision.visualization.render import render_frame


def make_white_frame(height: int = 140, width: int = 160) -> FrameData:
    return FrameData(
        frame_id=0,
        timestamp_ms=0.0,
        image=np.full((height, width, 3), 255, dtype=np.uint8),
    )


def make_box_observation(state: EntityState, entity_id: int = 1) -> EntityObservation:
    box = None if state is EntityState.LOST else BoundingBox(40.0, 40.0, 120.0, 120.0)
    return EntityObservation(
        entity_id=entity_id,
        state=state,
        bounding_box=box,
        class_label="person",
        frame_id=0,
    )


def make_zone() -> Zone:
    return Zone(
        name="lane",
        vertices=[(20.0, 20.0), (140.0, 20.0), (140.0, 80.0), (20.0, 80.0)],
    )


def make_open_event() -> Event:
    return Event(
        event_type=EventType.PROXIMITY_HAZARD,
        entity_ids=(1, 2),
        status=EventStatus.OPEN,
        opened_frame_id=5,
        closed_frame_id=None,
        zone_name=None,
    )


def make_closed_event() -> Event:
    return Event(
        event_type=EventType.ZONE_INTRUSION,
        entity_ids=(1,),
        status=EventStatus.CLOSED,
        opened_frame_id=5,
        closed_frame_id=12,
        zone_name="lane",
    )


class TestRenderImmutability:
    def test_returns_new_array_not_input(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [], [], [])
        assert output is not frame.image
        assert not np.shares_memory(output, frame.image)

    def test_returns_writeable_array(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [], [], [])
        assert output.flags.writeable

    def test_input_frame_image_provably_unmodified(self) -> None:
        frame = make_white_frame()
        original = frame.image.copy()
        render_frame(
            frame,
            [make_box_observation(EntityState.VISIBLE)],
            [make_zone()],
            [make_open_event()],
        )
        assert np.array_equal(frame.image, original)
        assert not frame.image.flags.writeable

    def test_empty_render_is_identical_copy(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [], [], [])
        assert np.array_equal(output, frame.image)


class TestEntityBoxes:
    def test_visible_box_drawn_solid_green(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [make_box_observation(EntityState.VISIBLE)], [], [])
        assert output[40, 80].tolist() == [0, 255, 0]
        assert output[80, 40].tolist() == [0, 255, 0]
        assert output[120, 100].tolist() == [0, 255, 0]
        assert output[100, 120].tolist() == [0, 255, 0]

    def test_occluded_box_drawn_solid_amber(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [make_box_observation(EntityState.OCCLUDED)], [], [])
        assert output[40, 80].tolist() == [255, 165, 0]

    def test_predicted_box_drawn_blue_dashed(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [make_box_observation(EntityState.PREDICTED)], [], [])
        # Top edge: dashes at x in [40,48), [52,60), ..., [112,120).
        assert output[40, 44].tolist() == [0, 0, 255]
        assert output[40, 116].tolist() == [0, 0, 255]
        assert output[40, 50].tolist() == [255, 255, 255]
        # Left edge: dashes at y in [40,48), [52,60), ...
        assert output[44, 40].tolist() == [0, 0, 255]
        assert output[50, 40].tolist() == [255, 255, 255]

    def test_lost_draws_nothing(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [make_box_observation(EntityState.LOST)], [], [])
        assert np.array_equal(output, frame.image)

    def test_retired_draws_nothing(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [make_box_observation(EntityState.RETIRED)], [], [])
        assert np.array_equal(output, frame.image)


class TestZones:
    def test_zone_outline_drawn(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [], [make_zone()], [])
        assert output[20, 80].tolist() == [255, 0, 255]
        assert output[50, 20].tolist() == [255, 0, 255]


class TestEventBanner:
    def test_open_event_draws_banner(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [], [], [make_open_event()])
        assert output[4, 10].tolist() == [0, 0, 0]
        assert np.any(np.all(output[:20, :] == [255, 255, 255], axis=2))

    def test_closed_event_draws_no_banner(self) -> None:
        frame = make_white_frame()
        output = render_frame(frame, [], [], [make_closed_event()])
        assert np.array_equal(output, frame.image)

    def test_zone_intrusion_open_banner_renders(self) -> None:
        event = Event(
            event_type=EventType.ZONE_INTRUSION,
            entity_ids=(7,),
            status=EventStatus.OPEN,
            opened_frame_id=3,
            closed_frame_id=None,
            zone_name="dock",
        )
        frame = make_white_frame()
        output = render_frame(frame, [], [], [event])
        assert output[4, 10].tolist() == [0, 0, 0]
