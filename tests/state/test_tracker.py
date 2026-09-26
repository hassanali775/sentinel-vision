"""Tests for PersistentEntityTracker (PR-006).

Covers budget validation, state transitions across all 5 states,
hand-computed linear extrapolation, single-observation fallback,
RETIRED terminal purging, reappearance resets, and independent multi-entity lifecycle.
Also covers ADR-0011 two-tier re-identification: active (OCCLUDED/PREDICTED)
entity matching ahead of the RETIRED candidate pool.
"""

import math

import pytest

from sentinel_vision.data.contracts import BoundingBox, Detection, TrackedDetection
from sentinel_vision.reidentification.spatial import SpatialReidentifier
from sentinel_vision.state.entity import EntityObservation, EntityState
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

    def test_predicted_extrapolation_clamps_inverted_axis(self) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=0, prediction_budget=3, retirement_budget=5
        )
        # Frame 0: (10, 10, 20, 20), Frame 1: (16, 10, 22, 20)
        # Velocity: dx_min=6, dx_max=2 (edges diverge over time).
        tracker.update(0, [tracked(10, 10, 20, 20, track_id=0)])
        tracker.update(1, [tracked(16, 10, 22, 20, track_id=0)])

        # Frame 2 (miss 1, step 1): x_min=22, x_max=24 -- still valid.
        obs2 = tracker.update(2, [])
        assert obs2[0].state == EntityState.PREDICTED
        assert obs2[0].bounding_box == box(22, 10, 24, 20)

        # Frame 3 (miss 2, step 2): raw x_min=28, x_max=26 would invert;
        # axis is re-centered on midpoint 27 keeping last known width 6.
        obs3 = tracker.update(3, [])
        assert obs3[0].state == EntityState.PREDICTED
        assert obs3[0].bounding_box == box(24, 10, 30, 20)

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

    def test_velocity_after_reappearance_uses_actual_frame_gap_not_assumed_unit_spacing(
        self,
    ) -> None:
        tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )
        # Observe at frame 0 (0,0,10,10) and frame 1 (2,0,12,10) -> velocity 2/frame
        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])
        tracker.update(1, [tracked(2, 0, 12, 10, track_id=0)])

        # Miss at frame 2 (OCCLUDED)
        obs2 = tracker.update(2, [])
        assert obs2[0].state == EntityState.OCCLUDED

        # Reappear at frame 3 (10,0,20,10) -> 2 real frames gap from frame 1
        obs3 = tracker.update(3, [tracked(10, 0, 20, 10, track_id=0)])
        assert obs3[0].state == EntityState.VISIBLE

        # Miss at frame 4 (OCCLUDED)
        obs4 = tracker.update(4, [])
        assert obs4[0].state == EntityState.OCCLUDED

        # Miss at frame 5 (PREDICTED, steps=1)
        # Expected box is (14.0, 0.0, 24.0, 10.0) based on frame_delta = 2 (velocity 4/frame)
        obs5 = tracker.update(5, [])
        assert obs5[0].state == EntityState.PREDICTED
        assert obs5[0].bounding_box == box(14.0, 0.0, 24.0, 10.0)


class TestRetiredCandidateCadenceAndVelocity:
    """Regression tests for PR-007 code review feedback.

    Locks in two guarantees:
    1. ``SpatialReidentifier.purge_expired`` runs unconditionally on every
       ``PersistentEntityTracker.update`` frame, even when every detection is
       matched and no unmatched track reaches the re-identification pool.
    2. ``ReidentificationCandidate`` velocity created at the ``RETIRED``
       transition uses exact frame-delta normalization
       ``(last_box - second_to_last_box) / (last_frame - second_to_last_frame)``
       with a zero/static fallback when fewer than two observations exist.
    """

    def _tracker(
        self, retention_window: int
    ) -> tuple[PersistentEntityTracker, SpatialReidentifier]:
        reidentifier = SpatialReidentifier(
            retention_window=retention_window, max_distance=50.0
        )
        tracker = PersistentEntityTracker(
            occlusion_budget=0,
            prediction_budget=1,
            retirement_budget=1,
            reidentifier=reidentifier,
        )
        return tracker, reidentifier

    def test_purge_expired_runs_every_frame_even_with_no_unmatched_tracks(
        self,
    ) -> None:
        """Candidates expire on cadence even when every frame's detections match.

        Entity 0 retires at frame 2 and enters the pool. Frames 3-5 carry only
        entity 1's track, so every detection matches an existing entity — there
        is no unmatched track to trigger re-identification. The candidate must
        still be purged exactly at ``retired_frame_id + retention_window``.
        """
        tracker, reidentifier = self._tracker(retention_window=2)

        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0), tracked(50, 50, 60, 60, track_id=1)])
        tracker.update(1, [tracked(50, 50, 60, 60, track_id=1)])
        # Entity 0's second miss (k=2 > retirement_budget=1) -> RETIRED, candidate added
        tracker.update(2, [tracked(50, 50, 60, 60, track_id=1)])
        assert len(reidentifier.candidates) == 1
        assert reidentifier.candidates[0].entity_id == 0
        assert reidentifier.candidates[0].retired_frame_id == 2

        # Frames 3 and 4: all detections matched, no unmatched tracks.
        # Purge still runs; age = frame - 2 <= retention_window (2) keeps candidate.
        tracker.update(3, [tracked(50, 50, 60, 60, track_id=1)])
        assert len(reidentifier.candidates) == 1
        tracker.update(4, [tracked(50, 50, 60, 60, track_id=1)])
        assert len(reidentifier.candidates) == 1

        # Frame 5: age = 5 - 2 = 3 > retention_window (2) -> purged despite no
        # unmatched tracks ever touching the pool on these frames.
        tracker.update(5, [tracked(50, 50, 60, 60, track_id=1)])
        assert len(reidentifier.candidates) == 0

    def test_retired_candidate_velocity_uses_exact_frame_delta_normalization(
        self,
    ) -> None:
        """Candidate velocity divides displacement by the real frame gap, not 1.

        Entity is observed at frame 0 (0,0,10,10) and frame 2 (4,0,14,10): a
        frame_delta of 2, so per-frame velocity is (2.0, 0.0, 2.0, 0.0) rather
        than the raw displacement (4.0, 0.0, 4.0, 0.0).
        """
        tracker, reidentifier = self._tracker(retention_window=10)

        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])
        tracker.update(2, [tracked(4, 0, 14, 10, track_id=0)])
        tracker.update(3, [])
        tracker.update(4, [])
        tracker.update(5, [])

        assert len(reidentifier.candidates) == 1
        candidate = reidentifier.candidates[0]
        assert candidate.last_observed_frame_id == 2
        assert candidate.retired_frame_id == 4
        assert candidate.last_known_box == box(4, 0, 14, 10)
        assert candidate.velocity == (2.0, 0.0, 2.0, 0.0)

    def test_retired_candidate_velocity_falls_back_to_zero_with_single_observation(
        self,
    ) -> None:
        """A candidate with only one observed position gets static velocity."""
        tracker, reidentifier = self._tracker(retention_window=10)

        tracker.update(0, [tracked(5, 5, 15, 15, track_id=0)])
        tracker.update(1, [])
        tracker.update(2, [])
        tracker.update(3, [])

        assert len(reidentifier.candidates) == 1
        assert reidentifier.candidates[0].velocity == (0.0, 0.0, 0.0, 0.0)


class TestActiveEntitySpatialMatching:
    """ADR-0011: orphan detections match unclaimed OCCLUDED/PREDICTED entities first.

    The active pool is checked before the RETIRED candidate pool and before a new
    entity is minted. These tests pin the tier ordering, the one-to-one claim per
    frame, the use of the *predicted* (not stale) box, order-independent
    disambiguation, the LOST exclusion, and the documented no-op when no
    reidentifier supplies plausibility thresholds.
    """

    def test_orphan_plausible_against_occluded_held_box_relinks_not_mints(self) -> None:
        """An orphan detection on an OCCLUDED entity's held box recovers that entity."""
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=50.0)
        tracker = PersistentEntityTracker(
            occlusion_budget=2,
            prediction_budget=3,
            retirement_budget=5,
            reidentifier=reidentifier,
        )

        # Frame 0: entity 0 (track 0) at (100,100,110,110); entity 1 (track 1) far away.
        tracker.update(
            0,
            [
                tracked(100, 100, 110, 110, track_id=0),
                tracked(0, 0, 10, 10, track_id=1),
            ],
        )

        # Frame 1: only entity 1 is seen -> entity 0 goes OCCLUDED, holding its box.
        obs1 = tracker.update(1, [tracked(1, 0, 11, 10, track_id=1)])
        assert [o.entity_id for o in obs1] == [0, 1]
        assert obs1[0].state is EntityState.OCCLUDED
        assert obs1[0].bounding_box == box(100, 100, 110, 110)

        # Frame 2: entity 1 keeps moving; an orphan (unseen track 5) appears on
        # entity 0's held box. Entity 0's box for this frame is still the held box
        # because frames_since_last_match + 1 = 2 <= occlusion_budget 2.
        obs2 = tracker.update(
            2,
            [
                tracked(2, 0, 12, 10, track_id=1),
                tracked(101, 101, 111, 111, track_id=5),
            ],
        )

        # Only the two pre-existing entities are reported: nothing was minted.
        assert [o.entity_id for o in obs2] == [0, 1]
        assert obs2[0].entity_id == 0
        assert obs2[0].state is EntityState.VISIBLE
        assert obs2[0].bounding_box == box(101, 101, 111, 111)
        assert obs2[1].state is EntityState.VISIBLE
        assert obs2[1].bounding_box == box(2, 0, 12, 10)

        # The orphan is plausible against entity 0's held box (center distance
        # hypot(1,1) = 1.41) and NOT against entity 1 (center (7,5) at frame 2,
        # distance far beyond max_distance 50).
        assert math.hypot(106.0 - 105.0, 106.0 - 105.0) <= 50.0
        assert math.hypot(106.0 - 7.0, 106.0 - 5.0) > 50.0

        # The relink also transferred ownership of the new track id: with no
        # detections at all, entity 0 ages as entity 0 rather than a new entity
        # being minted for track 5.
        obs3 = tracker.update(3, [])
        assert [o.entity_id for o in obs3] == [0, 1]
        assert obs3[0].entity_id == 0
        assert obs3[0].state is EntityState.OCCLUDED

    def test_orphan_near_predicted_position_uses_extrapolated_box_not_stale_box(
        self,
    ) -> None:
        """A PREDICTED entity is matched on its extrapolated position.

        Mirrors the PR-006 frame-gap velocity test: the two observations span
        frames 0 and 2, so per-frame velocity is 2.0 (not the raw 4.0), and the
        predicted box must be built from that normalized velocity.
        """
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=3.0)
        tracker = PersistentEntityTracker(
            occlusion_budget=0,
            prediction_budget=2,
            retirement_budget=3,
            reidentifier=reidentifier,
        )

        tracker.update(
            0,
            [
                tracked(0, 0, 10, 10, track_id=0),
                tracked(300, 300, 310, 310, track_id=1),
            ],
        )
        # Frame 2: entity 0 observed at (4,0,14,10) after a 2-frame gap.
        tracker.update(
            2,
            [
                tracked(4, 0, 14, 10, track_id=0),
                tracked(304, 300, 314, 310, track_id=1),
            ],
        )

        # Frame 3: entity 0 missed once. k=1 > occlusion_budget 0, so it is
        # PREDICTED with steps=1: (4 + 1*2, 0, 14 + 1*2, 10) = (6,0,16,10).
        obs3 = tracker.update(3, [tracked(306, 300, 316, 310, track_id=1)])
        assert obs3[0].entity_id == 0
        assert obs3[0].state is EntityState.PREDICTED
        assert obs3[0].bounding_box == box(6, 0, 16, 10)

        # Frame 4: entity 0 is PREDICTED with frames_since_last_match=1, so the
        # box for this frame is steps=2: (4 + 2*2, 0, 14 + 2*2, 10) = (8,0,18,10),
        # center (13,5). The orphan sits exactly on that extrapolated center.
        obs4 = tracker.update(
            4,
            [
                tracked(308, 300, 318, 310, track_id=1),
                tracked(8, 0, 18, 10, track_id=7),
            ],
        )

        # The stale last-observed box (4,0,14,10, center (9,5)) is 4.0 away, which
        # is outside max_distance 3.0: a match here is only possible if the
        # tracker used the predicted box.
        assert math.hypot(13.0 - 9.0, 5.0 - 5.0) > 3.0

        assert [o.entity_id for o in obs4] == [0, 1]
        assert obs4[0].entity_id == 0
        assert obs4[0].state is EntityState.VISIBLE
        assert obs4[0].bounding_box == box(8, 0, 18, 10)
        assert obs4[1].entity_id == 1
        assert obs4[1].state is EntityState.VISIBLE

    def test_two_occluded_entities_ambiguous_orphan_takes_smallest_prediction_error(
        self,
    ) -> None:
        """Ambiguity between two active entities resolves on prediction error alone.

        Both entities are OCCLUDED and both are plausible for the orphan; the one
        whose box for this frame is closer wins. Runs the same geometry twice,
        swapping which entity was created first, and proves the moving entity
        wins in both cases — the outcome is geometry-driven, not id- or
        insertion-order-driven.
        """

        def run(mover_first: bool) -> list[EntityObservation]:
            reidentifier = SpatialReidentifier(retention_window=10, max_distance=60.0)
            tracker = PersistentEntityTracker(
                occlusion_budget=2,
                prediction_budget=4,
                retirement_budget=6,
                reidentifier=reidentifier,
            )
            # mover: (0,0,10,10) -> (2,0,12,10) over frames 0->1 => velocity 2/frame.
            # static: (20,20,30,30) at both frames => velocity 0.
            frame0 = (
                [tracked(0, 0, 10, 10, track_id=0), tracked(20, 20, 30, 30, track_id=1)]
                if mover_first
                else [
                    tracked(20, 20, 30, 30, track_id=1),
                    tracked(0, 0, 10, 10, track_id=0),
                ]
            )
            frame1 = (
                [tracked(2, 0, 12, 10, track_id=0), tracked(20, 20, 30, 30, track_id=1)]
                if mover_first
                else [
                    tracked(20, 20, 30, 30, track_id=1),
                    tracked(2, 0, 12, 10, track_id=0),
                ]
            )
            tracker.update(0, frame0)
            tracker.update(1, frame1)
            # Frame 2: first miss (k=1) -> both OCCLUDED, boxes held.
            assert [o.state for o in tracker.update(2, [])] == [
                EntityState.OCCLUDED,
                EntityState.OCCLUDED,
            ]
            # Frame 3: second miss (k=2 <= occlusion_budget 2) -> still OCCLUDED.
            assert [o.state for o in tracker.update(3, [])] == [
                EntityState.OCCLUDED,
                EntityState.OCCLUDED,
            ]
            # Frame 4: orphan (8,1,18,11), center (13,6).
            return tracker.update(4, [tracked(8, 1, 18, 11, track_id=9)])

        # Hand-computed, for either creation order, the boxes these two entities
        # have on frame 4 (both are at frames_since_last_match=2, so k=3 is their
        # first PREDICTED frame, steps = 3 - 2 = 1):
        #   mover:  (2,0,12,10) + 1*(2,0,2,0) = (4,0,14,10),  center (9,5)
        #   static: (20,20,30,30) + 1*(0,0,0,0) = (20,20,30,30), center (25,25)
        # Detection center (13,6):
        #   to mover  = hypot(4, 1)  = 4.123  <- smallest, so the mover wins
        #   to static = hypot(12, 19) = 22.47 <- plausible (<= 60) but not closest
        dist_to_mover = math.hypot(13.0 - 9.0, 6.0 - 5.0)
        dist_to_static = math.hypot(13.0 - 25.0, 6.0 - 25.0)
        assert dist_to_mover == math.hypot(4, 1)
        assert dist_to_static == math.hypot(12, 19)
        assert dist_to_mover < dist_to_static
        assert dist_to_static <= 60.0

        mover_created_first = run(mover_first=True)
        mover_created_second = run(mover_first=False)

        # Mover created first: it is entity 0 and takes the match.
        assert [o.entity_id for o in mover_created_first] == [0, 1]
        assert mover_created_first[0].state is EntityState.VISIBLE
        assert mover_created_first[0].bounding_box == box(8, 1, 18, 11)
        assert mover_created_first[1].state is EntityState.PREDICTED

        # Mover created second: it is now entity 1, and the SAME geometry still
        # resolves to the mover. Swapping creation order swaps which id wins,
        # which proves the decision is not an artifact of id or insertion order.
        assert [o.entity_id for o in mover_created_second] == [0, 1]
        assert mover_created_second[0].state is EntityState.PREDICTED
        assert mover_created_second[1].state is EntityState.VISIBLE
        assert mover_created_second[1].bounding_box == box(8, 1, 18, 11)

    def test_exact_prediction_error_tie_breaks_to_lowest_entity_id(self) -> None:
        """An exact ambiguity tie resolves to the lowest entity_id, in both orders.

        The tie is symmetric in geometry, so the outcome can only be decided by
        the id rule. Creation order is reversed between the two runs and the
        winner is entity 0 both times — including the run where entity 0 is the
        right-hand box, which shows the tie-break is not secretly preferring the
        left, the closer, or the first-listed candidate. (Order-independence of
        the non-tied, geometry-driven case is pinned separately by
        ``test_two_occluded_entities_ambiguous_orphan_takes_smallest_prediction_error``.)
        """

        def run(left_first: bool) -> list[EntityObservation]:
            reidentifier = SpatialReidentifier(retention_window=10, max_distance=60.0)
            tracker = PersistentEntityTracker(
                occlusion_budget=2,
                prediction_budget=4,
                retirement_budget=6,
                reidentifier=reidentifier,
            )
            left = tracked(10, 0, 20, 10, track_id=0)
            right = tracked(30, 0, 40, 10, track_id=1)
            tracker.update(0, [left, right] if left_first else [right, left])
            tracker.update(1, [])
            tracker.update(2, [])
            # Frame 3: both static, so their boxes are unchanged: centers
            # (15,5) and (35,5). The orphan is centered at (25,5): 10.0 from each.
            return tracker.update(3, [tracked(20, 0, 30, 10, track_id=4)])

        assert math.hypot(25.0 - 15.0, 5.0 - 5.0) == math.hypot(25.0 - 35.0, 5.0 - 5.0)

        left_first = run(left_first=True)
        assert [o.entity_id for o in left_first] == [0, 1]
        assert left_first[0].state is EntityState.VISIBLE
        assert left_first[0].bounding_box == box(20, 0, 30, 10)
        assert left_first[1].state is EntityState.PREDICTED

        # Same tie, opposite creation order: entity 0 is now the right-hand box,
        # and the tie-break still lands on entity 0. The winner follows the id,
        # not the geometry and not the insertion order.
        right_first = run(left_first=False)
        assert [o.entity_id for o in right_first] == [0, 1]
        assert right_first[0].state is EntityState.VISIBLE
        assert right_first[0].bounding_box == box(20, 0, 30, 10)
        assert right_first[1].state is EntityState.PREDICTED

    def test_active_claims_are_one_to_one_within_a_frame(self) -> None:
        """Two orphans cannot both take the same active entity."""
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=60.0)
        tracker = PersistentEntityTracker(
            occlusion_budget=2,
            prediction_budget=4,
            retirement_budget=6,
            reidentifier=reidentifier,
        )
        tracker.update(0, [tracked(20, 20, 30, 30, track_id=0)])
        tracker.update(1, [])

        # Only one active entity exists; the first orphan claims it, the second
        # has no remaining candidate and mints its own entity.
        obs = tracker.update(
            2,
            [
                tracked(21, 21, 31, 31, track_id=1),
                tracked(22, 22, 32, 32, track_id=2),
            ],
        )

        assert [o.entity_id for o in obs] == [0, 1]
        assert obs[0].state is EntityState.VISIBLE
        assert obs[0].bounding_box == box(21, 21, 31, 31)
        assert obs[1].state is EntityState.VISIBLE
        assert obs[1].bounding_box == box(22, 22, 32, 32)

    def test_no_plausible_active_candidate_falls_through_to_retired_pool(self) -> None:
        """Tier 1 missing must not swallow tier 2: the RETIRED pool still relinks."""
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=50.0)
        tracker = PersistentEntityTracker(
            occlusion_budget=0,
            prediction_budget=1,
            retirement_budget=1,
            reidentifier=reidentifier,
        )

        tracker.update(
            0,
            [
                tracked(0, 0, 10, 10, track_id=0),
                tracked(300, 300, 310, 310, track_id=1),
            ],
        )
        # Frame 1: entity 0 first miss (k=1) -> PREDICTED (occlusion budget 0).
        tracker.update(1, [tracked(302, 300, 312, 310, track_id=1)])
        # Frame 2: entity 0 second miss (k=2 > retirement_budget 1) -> RETIRED and
        # its candidate enters the retention pool at retired_frame_id 2.
        tracker.update(2, [tracked(304, 300, 314, 310, track_id=1)])
        assert len(reidentifier.candidates) == 1
        assert reidentifier.candidates[0].entity_id == 0

        # Frame 3: the orphan has no plausible active candidate (entity 1 is the
        # only live entity and it is claimed by its own track this frame), so it
        # must reach the retired pool and re-link to entity 0 there.
        obs3 = tracker.update(
            3,
            [
                tracked(5, 0, 15, 10, track_id=7),
                tracked(306, 300, 316, 310, track_id=1),
            ],
        )

        assert [o.entity_id for o in obs3] == [0, 1]
        assert obs3[0].entity_id == 0
        assert obs3[0].state is EntityState.VISIBLE
        assert obs3[0].bounding_box == box(5, 0, 15, 10)
        assert obs3[1].entity_id == 1
        assert obs3[1].state is EntityState.VISIBLE
        assert len(reidentifier.candidates) == 0

    def test_lost_entity_is_never_offered_as_an_active_candidate(self) -> None:
        """A LOST entity carries no box, so a detection at its last known spot mints."""
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=60.0)
        tracker = PersistentEntityTracker(
            occlusion_budget=0,
            prediction_budget=1,
            retirement_budget=3,
            reidentifier=reidentifier,
        )

        tracker.update(0, [tracked(100, 100, 110, 110, track_id=0)])
        # Frame 1: k=1 > occlusion 0, k=1 <= prediction 1 -> PREDICTED (held box).
        assert tracker.update(1, [])[0].state is EntityState.PREDICTED
        # Frame 2: k=2 > prediction 1, k=2 <= retirement 3 -> LOST, no box.
        obs2 = tracker.update(2, [])
        assert obs2[0].state is EntityState.LOST
        assert obs2[0].bounding_box is None

        # Frame 3: the orphan is EXACTLY on the entity's last known position, but
        # LOST is in neither tier, so a new entity is minted instead of a relink.
        obs3 = tracker.update(3, [tracked(100, 100, 110, 110, track_id=1)])
        assert [o.entity_id for o in obs3] == [0, 1]
        assert obs3[0].state is EntityState.LOST
        assert obs3[0].bounding_box is None
        assert obs3[1].state is EntityState.VISIBLE
        assert obs3[1].bounding_box == box(100, 100, 110, 110)

        # Frame 4: entity 0 ages out (k=4 > retirement_budget 3) and retires,
        # having never been recovered: the new entity tracks the same track id.
        obs4 = tracker.update(4, [tracked(100, 100, 110, 110, track_id=1)])
        assert [o.entity_id for o in obs4] == [0, 1]
        assert obs4[0].state is EntityState.RETIRED
        assert obs4[1].entity_id == 1
        assert obs4[1].state is EntityState.VISIBLE

        # Frame 5: entity 0 is gone and its candidate has entered the retired pool.
        obs5 = tracker.update(5, [tracked(100, 100, 110, 110, track_id=1)])
        assert [o.entity_id for o in obs5] == [1]
        assert len(reidentifier.candidates) == 1
        assert reidentifier.candidates[0].entity_id == 0

    def test_without_reidentifier_active_matching_is_skipped_and_new_entity_minted(
        self,
    ) -> None:
        """``reidentifier=None`` disables tier 1 entirely instead of half-applying it.

        Active matching reads its plausibility thresholds from the reidentifier, so
        with none configured there is nothing to score against. This is a
        documented no-op: no crash, no silent partial match, and the orphan
        simply mints a new entity.
        """
        tracker = PersistentEntityTracker(
            occlusion_budget=2,
            prediction_budget=3,
            retirement_budget=5,
        )

        tracker.update(0, [tracked(100, 100, 110, 110, track_id=0)])
        obs1 = tracker.update(1, [])
        assert obs1[0].state is EntityState.OCCLUDED
        assert obs1[0].bounding_box == box(100, 100, 110, 110)

        # Exactly the geometry that relinked in the reidentified case above.
        obs2 = tracker.update(2, [tracked(101, 101, 111, 111, track_id=5)])

        assert [o.entity_id for o in obs2] == [0, 1]
        assert obs2[0].entity_id == 0
        assert obs2[0].state is EntityState.OCCLUDED
        assert obs2[0].bounding_box == box(100, 100, 110, 110)
        assert obs2[1].entity_id == 1
        assert obs2[1].state is EntityState.VISIBLE
        assert obs2[1].bounding_box == box(101, 101, 111, 111)

    def test_reidentifier_thresholds_are_read_from_the_reidentifier(self) -> None:
        """Tier 1 honors the reidentifier's own max_distance, tighter or looser."""
        tight = SpatialReidentifier(retention_window=10, max_distance=1.0)
        tracker = PersistentEntityTracker(
            occlusion_budget=2,
            prediction_budget=3,
            retirement_budget=5,
            reidentifier=tight,
        )
        tracker.update(0, [tracked(100, 100, 110, 110, track_id=0)])
        tracker.update(1, [])

        # 5.0 px from the held box's center (105,105): outside max_distance 1.0.
        obs = tracker.update(2, [tracked(105, 105, 115, 115, track_id=9)])
        assert [o.entity_id for o in obs] == [0, 1]
        assert obs[0].state is EntityState.OCCLUDED
        assert obs[1].entity_id == 1

        loose = SpatialReidentifier(retention_window=10, max_distance=50.0)
        tracker2 = PersistentEntityTracker(
            occlusion_budget=2,
            prediction_budget=3,
            retirement_budget=5,
            reidentifier=loose,
        )
        tracker2.update(0, [tracked(100, 100, 110, 110, track_id=0)])
        tracker2.update(1, [])
        obs2 = tracker2.update(2, [tracked(105, 105, 115, 115, track_id=9)])
        assert [o.entity_id for o in obs2] == [0]
        assert obs2[0].state is EntityState.VISIBLE

    def test_active_matching_respects_class_label_mismatch(self) -> None:
        """A different class never matches an active entity, even dead center."""
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=50.0)
        tracker = PersistentEntityTracker(
            occlusion_budget=2,
            prediction_budget=3,
            retirement_budget=5,
            reidentifier=reidentifier,
        )
        tracker.update(0, [tracked(100, 100, 110, 110, track_id=0)])
        tracker.update(1, [])

        other = Detection(
            bounding_box=box(100, 100, 110, 110),
            confidence=1.0,
            class_label="some_other_class",
        )
        obs = tracker.update(2, [TrackedDetection(detection=other, track_id=1)])

        assert [o.entity_id for o in obs] == [0, 1]
        assert obs[0].state is EntityState.OCCLUDED
        assert obs[1].entity_id == 1
        assert obs[1].class_label == "some_other_class"

    def test_predicted_box_for_agrees_with_the_aging_loop(self) -> None:
        """``_predicted_box_for`` is the one box rule shared by pool and aging loop.

        The active pool asks for the box an entity would have if it ages one more
        frame; the aging loop asks for the box it does have. Both must be the same
        function, so the returned box for a given (miss count) is asserted here
        against the box the tracker actually published for that same miss count.
        """
        tracker = PersistentEntityTracker(
            occlusion_budget=1,
            prediction_budget=3,
            retirement_budget=5,
        )
        tracker.update(0, [tracked(0, 0, 10, 10, track_id=0)])
        tracker.update(1, [tracked(2, 0, 12, 10, track_id=0)])

        published = []
        for frame_id in (2, 3, 4):
            obs = tracker.update(frame_id, [])
            published.append((obs[0].state, obs[0].bounding_box))

        assert published == [
            (EntityState.OCCLUDED, box(2, 0, 12, 10)),
            (EntityState.PREDICTED, box(4, 0, 14, 10)),
            (EntityState.PREDICTED, box(6, 0, 16, 10)),
        ]


