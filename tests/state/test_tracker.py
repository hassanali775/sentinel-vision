"""Tests for PersistentEntityTracker (PR-006).

Covers budget validation, state transitions across all 5 states,
hand-computed linear extrapolation, single-observation fallback,
RETIRED terminal purging, reappearance resets, and independent multi-entity lifecycle.
"""

import pytest

from sentinel_vision.data.contracts import BoundingBox, Detection, TrackedDetection
from sentinel_vision.state.entity import EntityState
from sentinel_vision.state.tracker import PersistentEntityTracker


def box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def tracked(
    x_min: float, y_min: float, x_max: float, y_max: float, track_id: int = 0
) -> TrackedDetection:
    det = Detection(
        bounding_box=box(x_min, y_min, x_max, y_max),
        confidence=1.0,
        class_label="synthetic_target",
    )
    return TrackedDetection(detection=det, track_id=track_id)


class TestPersistentEntityTracker:
    def test_constructor_budget_validation(self) -> None:
        with pytest.raises(ValueError, match="Budgets must satisfy"):
            PersistentEntityTracker(
                occlusion_budget=2, prediction_budget=1, retirement_budget=3
            )

        with pytest.raises(ValueError, match="Budgets must satisfy"):
            PersistentEntityTracker(
                occlusion_budget=1, prediction_budget=4, retirement_budget=3
            )

        with pytest.raises(ValueError, match="non-negative"):
            PersistentEntityTracker(
                occlusion_budget=-1, prediction_budget=2, retirement_budget=3
            )

        # Valid constructions
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )
        assert tracker is not None

    def test_new_track_creates_visible_entity(self) -> None:
        tracker = PersistentEntityTracker()
        obs = tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])
        assert len(obs) == 1
        assert obs[0].entity_id == 0
        assert obs[0].state == EntityState.VISIBLE
        assert obs[0].bounding_box == box(0, 0, 10, 10)
        assert obs[0].frame_id == 0

    def test_visible_to_occluded_transition(self) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )
        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])
        obs1 = tracker.update(1, [])
        assert len(obs1) == 1
        assert obs1[0].entity_id == 0
        assert obs1[0].state == EntityState.OCCLUDED
        assert obs1[0].bounding_box == box(0, 0, 10, 10)

    def test_occluded_to_predicted_transition_boundary(self) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )
        # Observed at frame 0 and 1
        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])
        tracker.update(1, [tracked(2, 0, 12, 10, track_id=0)])

        # Frame 2 (miss 1, k=1 <= occlusion_budget=1): OCCLUDED
        obs2 = tracker.update(2, [])
        assert obs2[0].state == EntityState.OCCLUDED

        # Frame 3 (miss 2, k=2 > occlusion_budget=1, k=2 <= prediction_budget=3): PREDICTED
        obs3 = tracker.update(3, [])
        assert obs3[0].state == EntityState.PREDICTED

    def test_predicted_linear_extrapolation(self) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )
        # Frame 0: (0, 0, 10, 10)
        # Frame 1: (2, 0, 12, 10) -> Velocity (dx_min=2, dy_min=0, dx_max=2, dy_max=0)
        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])
        tracker.update(1, [tracked(2, 0, 12, 10, track_id=0)])

        # Frame 2 (miss 1): OCCLUDED, box held at (2, 0, 12, 10)
        obs2 = tracker.update(2, [])
        assert obs2[0].bounding_box == box(2, 0, 12, 10)

        # Frame 3 (miss 2, step 1 of prediction): PREDICTED
        # Hand-computed: (2 + 1*2, 0, 12 + 1*2, 10) = (4, 0, 14, 10)
        obs3 = tracker.update(3, [])
        assert obs3[0].state == EntityState.PREDICTED
        assert obs3[0].bounding_box == box(4, 0, 14, 10)

        # Frame 4 (miss 3, step 2 of prediction): PREDICTED
        # Hand-computed: (2 + 2*2, 0, 12 + 2*2, 10) = (6, 0, 16, 10)
        obs4 = tracker.update(4, [])
        assert obs4[0].state == EntityState.PREDICTED
        assert obs4[0].bounding_box == box(6, 0, 16, 10)

    def test_single_observation_predicted_fallback(self) -> None:
        # Single observation before miss: no second-to-last box exists
        tracker = PersistentEntityTracker(
            occlusion_budget=0, prediction_budget=2, retirement_budget=3
        )
        tracker.update(0, [tracked(5, 5, 15, 15, track_id=0)])

        # Frame 1 (miss 1, k=1 > occlusion_budget=0): PREDICTED
        # Fallback must hold last known box (5, 5, 15, 15)
        obs1 = tracker.update(1, [])
        assert obs1[0].state == EntityState.PREDICTED
        assert obs1[0].bounding_box == box(5, 5, 15, 15)

    def test_predicted_to_lost_transition_boundary(self) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=2, retirement_budget=4
        )
        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])

        # miss 1 (k=1 <= 1): OCCLUDED
        assert tracker.update(1, [])[0].state == EntityState.OCCLUDED
        # miss 2 (k=2 <= 2): PREDICTED
        assert tracker.update(2, [])[0].state == EntityState.PREDICTED
        # miss 3 (k=3 > 2, k=3 <= 4): LOST
        obs3 = tracker.update(3, [])
        assert obs3[0].state == EntityState.LOST
        assert obs3[0].bounding_box is None

    def test_lost_to_retired_transition_boundary(self) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=2, retirement_budget=3
        )
        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])

        # miss 1 (k=1): OCCLUDED
        tracker.update(1, [])
        # miss 2 (k=2): PREDICTED
        tracker.update(2, [])
        # miss 3 (k=3): LOST
        tracker.update(3, [])

        # miss 4 (k=4 > retirement_budget=3): RETIRED
        obs4 = tracker.update(4, [])
        assert len(obs4) == 1
        assert obs4[0].entity_id == 0
        assert obs4[0].state == EntityState.RETIRED
        assert obs4[0].bounding_box == box(0, 0, 10, 10)

        # Subsequent updates (miss 5+): entity is absent from return value
        obs5 = tracker.update(5, [])
        assert len(obs5) == 0

        obs6 = tracker.update(6, [])
        assert len(obs6) == 0

    @pytest.mark.parametrize(
        "miss_frames, expected_intermediate_state",
        [
            (1, EntityState.OCCLUDED),
            (2, EntityState.PREDICTED),
            (4, EntityState.LOST),
        ],
    )
    def test_reappearance_resets_to_visible(
        self, miss_frames: int, expected_intermediate_state: EntityState
    ) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )
        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])
        tracker.update(1, [tracked(2, 0, 12, 10, track_id=0)])

        current_frame = 2
        for _ in range(miss_frames):
            obs = tracker.update(current_frame, [])
            current_frame += 1

        assert obs[0].state == expected_intermediate_state

        # Reappearance with new detection at (50, 50, 60, 60)
        reappear_obs = tracker.update(
            current_frame, [tracked(50, 50, 60, 60, track_id=0)]
        )
        assert len(reappear_obs) == 1
        assert reappear_obs[0].entity_id == 0
        assert reappear_obs[0].state == EntityState.VISIBLE
        assert reappear_obs[0].bounding_box == box(50, 50, 60, 60)

    def test_multiple_simultaneous_entities(self) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )
        # Frame 0: entity 0 and entity 1 present
        tracker.update(
            0, [tracked(0, 0, 10, 10, track_id=0), tracked(20, 20, 30, 30, track_id=1)]
        )

        # Frame 1: entity 0 present, entity 1 missed (miss 1 -> OCCLUDED)
        obs1 = tracker.update(1, [tracked(1, 0, 11, 10, track_id=0)])
        assert len(obs1) == 2
        assert obs1[0].entity_id == 0
        assert obs1[0].state == EntityState.VISIBLE
        assert obs1[1].entity_id == 1
        assert obs1[1].state == EntityState.OCCLUDED

        # Frame 2: entity 0 present, entity 1 missed (miss 2 -> PREDICTED)
        obs2 = tracker.update(2, [tracked(2, 0, 12, 10, track_id=0)])
        assert len(obs2) == 2
        assert obs2[0].entity_id == 0
        assert obs2[0].state == EntityState.VISIBLE
        assert obs2[1].entity_id == 1
        assert obs2[1].state == EntityState.PREDICTED
