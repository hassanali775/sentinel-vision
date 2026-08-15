"""Tests for the stateless event rules (PR-009).

Rules are per-frame predicates: given a ``SpatialFrameObservation``, return
the set of keys whose condition holds. These tests pin the exact key
conventions and the strict threshold boundary so the integration test's
hand-computed frame numbers remain meaningful.
"""

import pytest

from sentinel_vision.events.event import EventType
from sentinel_vision.events.rules import BaseEventRule, ProximityHazardRule, ZoneIntrusionRule
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


class TestProximityHazardRule:
    def test_holds_for_pairs_strictly_below_threshold(self) -> None:
        rule = ProximityHazardRule(threshold_px=10.0)
        spatial = _spatial(
            frame_id=0,
            pairwise_distances={(0, 1): 5.0, (1, 2): 9.99},
        )
        assert rule.condition_holds(spatial) == {(0, 1), (1, 2)}

    def test_pair_at_threshold_is_not_a_hazard(self) -> None:
        rule = ProximityHazardRule(threshold_px=10.0)
        spatial = _spatial(frame_id=0, pairwise_distances={(0, 1): 10.0})
        assert rule.condition_holds(spatial) == set()

    def test_empty_frame_holds_for_nothing(self) -> None:
        rule = ProximityHazardRule(threshold_px=10.0)
        assert rule.condition_holds(_spatial(frame_id=0)) == set()

    def test_event_type_and_zone_name(self) -> None:
        rule = ProximityHazardRule(threshold_px=10.0)
        assert rule.event_type is EventType.PROXIMITY_HAZARD
        assert rule.zone_name is None

    @pytest.mark.parametrize("bad_threshold", [0.0, -1.0])
    def test_rejects_non_positive_threshold(self, bad_threshold: float) -> None:
        with pytest.raises(ValueError, match="must be > 0.0"):
            ProximityHazardRule(threshold_px=bad_threshold)


class TestZoneIntrusionRule:
    def test_holds_for_entities_in_named_zone(self) -> None:
        rule = ZoneIntrusionRule(zone_name="lane_a")
        spatial = _spatial(
            frame_id=0,
            zone_memberships={0: ["lane_a"], 1: [], 2: ["lane_b", "lane_a"]},
        )
        assert rule.condition_holds(spatial) == {0, 2}

    def test_holds_for_nothing_when_zone_absent(self) -> None:
        rule = ZoneIntrusionRule(zone_name="lane_a")
        spatial = _spatial(frame_id=0, zone_memberships={0: ["lane_b"], 1: []})
        assert rule.condition_holds(spatial) == set()

    def test_event_type_and_zone_name(self) -> None:
        rule = ZoneIntrusionRule(zone_name="lane_a")
        assert rule.event_type is EventType.ZONE_INTRUSION
        assert rule.zone_name == "lane_a"

    @pytest.mark.parametrize("bad_name", ["", "   "])
    def test_rejects_blank_zone_name(self, bad_name: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ZoneIntrusionRule(zone_name=bad_name)


class TestBaseEventRule:
    def test_cannot_instantiate_without_condition_holds(self) -> None:
        class BareRule(BaseEventRule):
            pass

        with pytest.raises(TypeError, match="abstract"):
            BareRule()
