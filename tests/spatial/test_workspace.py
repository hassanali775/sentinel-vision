"""Tests for the spatial workspace model (PR-008).

Covers zone membership (single, zero, and multiple overlapping zones),
exclusion of LOST entities from all spatial facts with a hand-computed
assertion, hand-computed pairwise distance, order-independent distance
keying, duplicate zone name rejection, and eligibility of OCCLUDED/PREDICTED
states with exclusion of RETIRED.
"""

import pytest

from sentinel_vision.data.contracts import BoundingBox
from sentinel_vision.spatial.workspace import SpatialFrameObservation, WorkspaceModel
from sentinel_vision.spatial.zone import Zone
from sentinel_vision.state.entity import EntityObservation, EntityState


def box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def obs(
    entity_id: int,
    state: EntityState,
    bounding_box: BoundingBox | None,
    frame_id: int = 0,
) -> EntityObservation:
    return EntityObservation(
        entity_id=entity_id,
        state=state,
        bounding_box=bounding_box,
        class_label="synthetic_target",
        frame_id=frame_id,
    )


def zone(name: str, vertices: list[tuple[float, float]]) -> Zone:
    return Zone(name=name, vertices=vertices)


def unit_square(name: str = "zone_a") -> Zone:
    return zone(name, [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])


class TestWorkspaceModel:
    def test_single_entity_in_one_zone(self) -> None:
        model = WorkspaceModel(zones=[unit_square()])
        result = model.evaluate(
            0, [obs(entity_id=0, state=EntityState.VISIBLE, bounding_box=box(2, 2, 8, 8))]
        )
        assert result.zone_memberships == {0: ["zone_a"]}
        assert result.pairwise_distances == {}

    def test_entity_in_zero_zones(self) -> None:
        model = WorkspaceModel(zones=[unit_square()])
        result = model.evaluate(
            0,
            [obs(entity_id=7, state=EntityState.VISIBLE, bounding_box=box(20, 20, 30, 30))],
        )
        # Eligible entity with its center in no zone maps to an empty list.
        assert result.zone_memberships == {7: []}
        assert result.pairwise_distances == {}

    def test_entity_in_two_overlapping_zones_simultaneously(self) -> None:
        zone_a = unit_square("zone_a")
        zone_b = zone("zone_b", [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)])
        model = WorkspaceModel(zones=[zone_a, zone_b])
        # Center (7,7) lies inside both overlapping squares.
        result = model.evaluate(
            0, [obs(entity_id=3, state=EntityState.VISIBLE, bounding_box=box(5, 5, 9, 9))]
        )
        assert result.zone_memberships == {3: ["zone_a", "zone_b"]}
        assert result.pairwise_distances == {}

    def test_lost_entity_excluded_from_all_spatial_facts(self) -> None:
        """Hand-computed: the LOST entity must appear nowhere.

        Entity 0 visible at box (0,0,10,10) -> center (5,5), inside zone_a.
        Entity 1 visible at box (10,0,20,10) -> center (15,5), outside zone_a.
        Entity 2 LOST -> no box, no facts.
        Distance between centers (5,5) and (15,5) is exactly 10.0.
        """
        model = WorkspaceModel(zones=[unit_square()])
        result = model.evaluate(
            0,
            [
                obs(entity_id=0, state=EntityState.VISIBLE, bounding_box=box(0, 0, 10, 10)),
                obs(entity_id=1, state=EntityState.VISIBLE, bounding_box=box(10, 0, 20, 10)),
                obs(entity_id=2, state=EntityState.LOST, bounding_box=None),
            ],
        )
        assert 2 not in result.zone_memberships
        assert result.zone_memberships == {0: ["zone_a"], 1: []}
        assert result.pairwise_distances == {(0, 1): 10.0}
        assert (0, 2) not in result.pairwise_distances
        assert (1, 2) not in result.pairwise_distances

    def test_pairwise_distance_hand_computed(self) -> None:
        model = WorkspaceModel(zones=[])
        result = model.evaluate(
            0,
            [
                obs(entity_id=0, state=EntityState.VISIBLE, bounding_box=box(0, 0, 10, 10)),
                obs(entity_id=1, state=EntityState.VISIBLE, bounding_box=box(3, 4, 13, 14)),
            ],
        )
        # Centers (5,5) and (8,9): distance sqrt(3^2 + 4^2) = 5.0
        assert result.pairwise_distances == {(0, 1): pytest.approx(5.0)}

    def test_distance_key_ordering_is_order_independent(self) -> None:
        model = WorkspaceModel(zones=[])
        entity_a = obs(entity_id=10, state=EntityState.VISIBLE, bounding_box=box(0, 0, 10, 10))
        entity_b = obs(entity_id=5, state=EntityState.VISIBLE, bounding_box=box(0, 10, 10, 20))
        result_ab = model.evaluate(0, [entity_a, entity_b])
        result_ba = model.evaluate(0, [entity_b, entity_a])
        # Key is always (min_id, max_id) = (5, 10); value is 10.0 either way.
        assert result_ab.pairwise_distances == {(5, 10): pytest.approx(10.0)}
        assert result_ab.pairwise_distances == result_ba.pairwise_distances

    def test_duplicate_zone_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate zone name"):
            WorkspaceModel(
                zones=[
                    unit_square("zone_a"),
                    zone("zone_a", [(20.0, 20.0), (30.0, 20.0), (30.0, 30.0)]),
                ]
            )

    def test_occluded_and_predicted_entities_contribute_facts(self) -> None:
        model = WorkspaceModel(zones=[unit_square()])
        result = model.evaluate(
            0,
            [
                obs(entity_id=0, state=EntityState.OCCLUDED, bounding_box=box(0, 0, 10, 10)),
                obs(entity_id=1, state=EntityState.PREDICTED, bounding_box=box(2, 2, 6, 6)),
            ],
        )
        assert result.zone_memberships == {0: ["zone_a"], 1: ["zone_a"]}
        assert (0, 1) in result.pairwise_distances

    def test_retired_entity_excluded_despite_final_box(self) -> None:
        model = WorkspaceModel(zones=[unit_square()])
        result = model.evaluate(
            0,
            [
                obs(entity_id=4, state=EntityState.RETIRED, bounding_box=box(0, 0, 10, 10)),
                obs(entity_id=5, state=EntityState.VISIBLE, bounding_box=box(20, 20, 30, 30)),
            ],
        )
        assert 4 not in result.zone_memberships
        assert result.zone_memberships == {5: []}
        assert result.pairwise_distances == {}

    def test_empty_observation_list_yields_empty_facts(self) -> None:
        model = WorkspaceModel(zones=[unit_square()])
        result = model.evaluate(0, [])
        assert result.zone_memberships == {}
        assert result.pairwise_distances == {}

    def test_evaluate_rejects_negative_frame_id(self) -> None:
        model = WorkspaceModel(zones=[])
        with pytest.raises(ValueError, match="frame_id"):
            model.evaluate(-1, [])

    def test_spatial_frame_observation_rejects_negative_frame_id(self) -> None:
        with pytest.raises(ValueError, match="frame_id"):
            SpatialFrameObservation(
                frame_id=-1, zone_memberships={}, pairwise_distances={}
            )
