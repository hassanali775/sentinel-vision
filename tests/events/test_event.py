"""Contract tests for the Event dataclass (PR-009).

Every invariant in ``Event.__post_init__`` is exercised here: the entity-count
and zone_name shape per event type, non-negative ids and frame ids, and the
open/closed frame ordering. A malformed event is a data-quality bug and must
be rejected at construction.
"""

import pytest

from sentinel_vision.events.event import Event, EventStatus, EventType


class TestProximityHazardEvent:
    def test_valid_open_event(self) -> None:
        event = Event(
            event_type=EventType.PROXIMITY_HAZARD,
            entity_ids=(0, 1),
            status=EventStatus.OPEN,
            opened_frame_id=5,
            closed_frame_id=None,
            zone_name=None,
        )
        assert event.entity_ids == (0, 1)
        assert event.zone_name is None

    def test_valid_closed_event(self) -> None:
        event = Event(
            event_type=EventType.PROXIMITY_HAZARD,
            entity_ids=(0, 1),
            status=EventStatus.CLOSED,
            opened_frame_id=5,
            closed_frame_id=9,
            zone_name=None,
        )
        assert event.status is EventStatus.CLOSED
        assert event.closed_frame_id == 9

    def test_rejects_wrong_entity_count(self) -> None:
        with pytest.raises(ValueError, match="exactly two entities"):
            Event(
                event_type=EventType.PROXIMITY_HAZARD,
                entity_ids=(0,),
                status=EventStatus.OPEN,
                opened_frame_id=0,
                closed_frame_id=None,
                zone_name=None,
            )

    def test_rejects_zone_name(self) -> None:
        with pytest.raises(ValueError, match="must not carry a zone_name"):
            Event(
                event_type=EventType.PROXIMITY_HAZARD,
                entity_ids=(0, 1),
                status=EventStatus.OPEN,
                opened_frame_id=0,
                closed_frame_id=None,
                zone_name="lane_a",
            )

    def test_rejects_negative_entity_id(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            Event(
                event_type=EventType.PROXIMITY_HAZARD,
                entity_ids=(-1, 1),
                status=EventStatus.OPEN,
                opened_frame_id=0,
                closed_frame_id=None,
                zone_name=None,
            )


class TestZoneIntrusionEvent:
    def test_valid_open_event(self) -> None:
        event = Event(
            event_type=EventType.ZONE_INTRUSION,
            entity_ids=(2,),
            status=EventStatus.OPEN,
            opened_frame_id=3,
            closed_frame_id=None,
            zone_name="lane_a",
        )
        assert event.entity_ids == (2,)
        assert event.zone_name == "lane_a"

    def test_rejects_missing_zone_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty zone_name"):
            Event(
                event_type=EventType.ZONE_INTRUSION,
                entity_ids=(2,),
                status=EventStatus.OPEN,
                opened_frame_id=0,
                closed_frame_id=None,
                zone_name=None,
            )

    def test_rejects_blank_zone_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty zone_name"):
            Event(
                event_type=EventType.ZONE_INTRUSION,
                entity_ids=(2,),
                status=EventStatus.OPEN,
                opened_frame_id=0,
                closed_frame_id=None,
                zone_name="   ",
            )

    def test_rejects_wrong_entity_count(self) -> None:
        with pytest.raises(ValueError, match="exactly one entity"):
            Event(
                event_type=EventType.ZONE_INTRUSION,
                entity_ids=(2, 3),
                status=EventStatus.OPEN,
                opened_frame_id=0,
                closed_frame_id=None,
                zone_name="lane_a",
            )


class TestFrameIdsAndStatus:
    def test_rejects_negative_opened_frame_id(self) -> None:
        with pytest.raises(ValueError, match="opened_frame_id"):
            Event(
                event_type=EventType.ZONE_INTRUSION,
                entity_ids=(2,),
                status=EventStatus.OPEN,
                opened_frame_id=-1,
                closed_frame_id=None,
                zone_name="lane_a",
            )

    def test_rejects_closed_frame_id_on_open_event(self) -> None:
        with pytest.raises(ValueError, match="must be None when status is OPEN"):
            Event(
                event_type=EventType.ZONE_INTRUSION,
                entity_ids=(2,),
                status=EventStatus.OPEN,
                opened_frame_id=0,
                closed_frame_id=4,
                zone_name="lane_a",
            )

    def test_rejects_missing_closed_frame_id_on_closed_event(self) -> None:
        with pytest.raises(ValueError, match="cannot be None when status is CLOSED"):
            Event(
                event_type=EventType.ZONE_INTRUSION,
                entity_ids=(2,),
                status=EventStatus.CLOSED,
                opened_frame_id=0,
                closed_frame_id=None,
                zone_name="lane_a",
            )

    def test_rejects_closed_frame_id_equal_to_opened(self) -> None:
        with pytest.raises(ValueError, match="must be > opened_frame_id"):
            Event(
                event_type=EventType.ZONE_INTRUSION,
                entity_ids=(2,),
                status=EventStatus.CLOSED,
                opened_frame_id=4,
                closed_frame_id=4,
                zone_name="lane_a",
            )

    def test_rejects_closed_frame_id_before_opened(self) -> None:
        with pytest.raises(ValueError, match="must be > opened_frame_id"):
            Event(
                event_type=EventType.ZONE_INTRUSION,
                entity_ids=(2,),
                status=EventStatus.CLOSED,
                opened_frame_id=5,
                closed_frame_id=3,
                zone_name="lane_a",
            )
