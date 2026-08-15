"""Hysteresis behavior tests for EventEngine (PR-009).

These tests drive the engine with hand-constructed ``SpatialFrameObservation``
values whose truth values are trivial to control, so the sustain/clear
semantics — including the flicker guarantee and fresh clear budget after
reopening — are pinned exactly, independently of the full pipeline.
"""

import pytest

from sentinel_vision.events.engine import EventEngine
from sentinel_vision.events.event import EventStatus, EventType
from sentinel_vision.events.rules import ProximityHazardRule, ZoneIntrusionRule
from sentinel_vision.spatial.workspace import SpatialFrameObservation


def _spatial(
    frame_id: int,
    zone_memberships: dict[int, list[str]] | None = None,
    pairwise_distances: dict[tuple[int, int], float] | None = None,
) -> SpatialFrameObservation:
    return SpatialFrameObservation(
        frame_id=frame_id,
        zone_memberships=zone_memberships or {},
        pairwise_distances=pairwise_distances or {},
    )


def _near(frame_id: int, pairs: list[tuple[int, int]]) -> SpatialFrameObservation:
    """Spatial frame where exactly ``pairs`` are close (distance 5.0 < 10.0)."""
    return _spatial(
        frame_id=frame_id,
        pairwise_distances={pair: 5.0 for pair in pairs},
    )


class TestSustain:
    def test_opens_after_sustain_frames(self) -> None:
        engine = EventEngine([(ProximityHazardRule(threshold_px=10.0), 3, 1)])
        near = _near(0, [(0, 1)])

        assert engine.update(0, near) == []
        assert engine.update(1, near) == []
        opened = engine.update(2, near)
        assert len(opened) == 1
        assert opened[0].event_type is EventType.PROXIMITY_HAZARD
        assert opened[0].entity_ids == (0, 1)
        assert opened[0].status is EventStatus.OPEN
        assert opened[0].opened_frame_id == 2

        assert engine.update(3, near) == []

    def test_interrupted_sustain_streak_restarts(self) -> None:
        engine = EventEngine([(ProximityHazardRule(threshold_px=10.0), 3, 1)])
        near = _near(0, [(0, 1)])

        assert engine.update(0, near) == []
        assert engine.update(1, _near(1, [])) == []
        assert engine.update(2, near) == []
        assert engine.update(3, near) == []
        opened = engine.update(4, near)
        assert len(opened) == 1
        assert opened[0].opened_frame_id == 4


class TestClear:
    def test_closes_after_clear_frames(self) -> None:
        engine = EventEngine([(ProximityHazardRule(threshold_px=10.0), 1, 2)])
        near = _near(0, [(0, 1)])
        far = _near(1, [])

        opened = engine.update(0, near)
        assert opened[0].status is EventStatus.OPEN
        assert engine.update(1, near) == []

        assert engine.update(2, far) == []
        closed = engine.update(3, far)
        assert len(closed) == 1
        assert closed[0].status is EventStatus.CLOSED
        assert closed[0].opened_frame_id == 0
        assert closed[0].closed_frame_id == 3

    def test_open_event_survives_brief_false_blip(self) -> None:
        engine = EventEngine([(ProximityHazardRule(threshold_px=10.0), 1, 3)])
        near = _near(0, [(0, 1)])
        far = _near(0, [])

        opened = engine.update(0, near)
        assert opened[0].status is EventStatus.OPEN
        assert engine.update(1, far) == []
        assert engine.update(2, far) == []
        assert engine.update(3, near) == []
        assert engine.update(4, near) == []
        assert engine.update(5, far) == []
        assert engine.update(6, far) == []
        closed = engine.update(7, far)
        assert len(closed) == 1
        assert closed[0].status is EventStatus.CLOSED
        assert closed[0].opened_frame_id == 0
        assert closed[0].closed_frame_id == 7

    def test_reopening_starts_fresh_clear_budget(self) -> None:
        engine = EventEngine([(ProximityHazardRule(threshold_px=10.0), 1, 2)])
        near = _near(0, [(0, 1)])
        far = _near(0, [])

        opened = engine.update(0, near)
        assert opened[0].status is EventStatus.OPEN
        assert engine.update(1, far) == []
        closed = engine.update(2, far)
        assert closed[0].status is EventStatus.CLOSED
        assert closed[0].opened_frame_id == 0

        assert engine.update(3, far) == []
        reopened = engine.update(4, near)
        assert reopened[0].status is EventStatus.OPEN
        assert reopened[0].opened_frame_id == 4
        assert reopened[0].closed_frame_id is None

        assert engine.update(5, far) == []
        reclosed = engine.update(6, far)
        assert len(reclosed) == 1
        assert reclosed[0].status is EventStatus.CLOSED
        assert reclosed[0].opened_frame_id == 4
        assert reclosed[0].closed_frame_id == 6


class TestMultipleKeys:
    def test_keys_are_tracked_independently(self) -> None:
        engine = EventEngine([(ZoneIntrusionRule(zone_name="lane_a"), 2, 1)])

        def members(frame_id: int, *entity_ids: int) -> SpatialFrameObservation:
            return _spatial(
                frame_id=frame_id,
                zone_memberships={
                    entity_id: ["lane_a"] for entity_id in entity_ids
                },
            )

        assert engine.update(0, members(0, 0)) == []
        first_open = engine.update(1, members(1, 0, 1))
        assert len(first_open) == 1
        assert first_open[0].entity_ids == (0,)
        assert first_open[0].opened_frame_id == 1

        second_open = engine.update(2, members(2, 0, 1))
        assert len(second_open) == 1
        assert second_open[0].entity_ids == (1,)
        assert second_open[0].opened_frame_id == 2

        closes = engine.update(3, members(3))
        assert [e.entity_ids for e in closes] == [(0,), (1,)]
        assert all(e.status is EventStatus.CLOSED for e in closes)
        assert all(e.closed_frame_id == 3 for e in closes)

    def test_closed_events_are_emitted_in_sorted_key_order(self) -> None:
        engine = EventEngine([(ZoneIntrusionRule(zone_name="lane_a"), 1, 1)])
        both = _spatial(
            frame_id=0, zone_memberships={0: ["lane_a"], 5: ["lane_a"]}
        )
        engine.update(0, both)
        closes = engine.update(1, _spatial(frame_id=1))
        assert [e.entity_ids for e in closes] == [(0,), (5,)]


class TestOrdering:
    def test_rules_evaluated_in_registration_order(self) -> None:
        engine = EventEngine(
            [
                (ZoneIntrusionRule(zone_name="lane_a"), 1, 1),
                (ProximityHazardRule(threshold_px=10.0), 1, 1),
            ]
        )
        spatial = _spatial(
            frame_id=0,
            zone_memberships={0: ["lane_a"]},
            pairwise_distances={(0, 1): 5.0},
        )
        events = engine.update(0, spatial)
        assert [e.event_type for e in events] == [
            EventType.ZONE_INTRUSION,
            EventType.PROXIMITY_HAZARD,
        ]
        assert events[0].zone_name == "lane_a"
        assert events[1].zone_name is None


class TestValidation:
    def test_constructor_rejects_empty_rule_list(self) -> None:
        with pytest.raises(ValueError, match="at least one rule"):
            EventEngine([])

    @pytest.mark.parametrize("bad_sustain", [0, -1])
    def test_constructor_rejects_invalid_sustain(self, bad_sustain: int) -> None:
        rule = ProximityHazardRule(threshold_px=10.0)
        with pytest.raises(ValueError, match="sustain_frames"):
            EventEngine([(rule, bad_sustain, 1)])

    @pytest.mark.parametrize("bad_clear", [0, -1])
    def test_constructor_rejects_invalid_clear(self, bad_clear: int) -> None:
        rule = ProximityHazardRule(threshold_px=10.0)
        with pytest.raises(ValueError, match="clear_frames"):
            EventEngine([(rule, 1, bad_clear)])

    def test_update_rejects_negative_frame_id(self) -> None:
        engine = EventEngine([(ProximityHazardRule(threshold_px=10.0), 1, 1)])
        with pytest.raises(ValueError, match="frame_id"):
            engine.update(-1, _near(0, [(0, 1)]))
