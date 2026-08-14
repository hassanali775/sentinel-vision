"""Integration tests for full pipeline with multi-object re-identification (PR-007).

Tests full pipeline (MultiObjectSyntheticFrameStream -> SyntheticBoxDetector ->
GreedyIoUTracker -> PersistentEntityTracker with SpatialReidentifier) with multiple objects,
where one object is occluded long enough to retire, reappears, and is correctly re-linked
to its original entity_id rather than getting a fresh entity_id.
"""

from sentinel_vision.detection.synthetic import SyntheticBoxDetector
from sentinel_vision.ingestion.stream import (
    MultiObjectSyntheticFrameStream,
    SyntheticObjectConfig,
)
from sentinel_vision.reidentification.spatial import SpatialReidentifier
from sentinel_vision.state.entity import EntityObservation, EntityState
from sentinel_vision.state.tracker import PersistentEntityTracker
from sentinel_vision.tracking.greedy import GreedyIoUTracker


class TestReidentificationPipeline:
    def test_full_pipeline_multi_object_reidentification(self) -> None:
        """Full pipeline multi-object spatial re-identification test.

        Object 0 moves continuously across all frames.
        Object 1 is visible on frames 0-2, occluded/absent on frames 3-7 (5 gap frames,
        exceeding retirement_budget 3), causing it to transition to RETIRED on frame 7 and
        enter the SpatialReidentifier retention pool.
        On frame 8, Object 1 reappears. GreedyIoUTracker (max_age=1) drops its track_id and
        issues a new track_id. PersistentEntityTracker queries SpatialReidentifier, matches
        the reappearing detection to Candidate 1's predicted trajectory, and re-links it to
        its original entity_id=1.
        """
        # Object 0: moves continuously along y=10
        obj0 = SyntheticObjectConfig(
            start_x=10.0,
            start_y=10.0,
            velocity_x=2.0,
            velocity_y=0.0,
            width=10,
            height=10,
            active_frames=range(0, 15),
        )

        # Object 1: moves along y=50, active 0..2, inactive 3..8, active 9..14
        active_frames_obj1 = set(range(0, 3)).union(set(range(9, 15)))
        obj1 = SyntheticObjectConfig(
            start_x=10.0,
            start_y=50.0,
            velocity_x=2.0,
            velocity_y=0.0,
            width=10,
            height=10,
            active_frames=active_frames_obj1,
        )

        stream = MultiObjectSyntheticFrameStream(
            width=200,
            height=100,
            fps=10.0,
            num_frames=13,
            objects=[obj0, obj1],
        )

        detector = SyntheticBoxDetector(multi_instance=True)
        # Upstream tracker has max_age=1 (shorter than occlusion gap of 6 frames)
        tracker = GreedyIoUTracker(iou_threshold=0.5, max_age=1)
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=30.0)
        entity_tracker = PersistentEntityTracker(
            occlusion_budget=1,
            prediction_budget=3,
            retirement_budget=5,
            reidentifier=reidentifier,
        )

        all_observations: list[list[EntityObservation]] = []

        for frame in stream:
            detections = detector.detect(frame)
            tracked = tracker.track(frame, detections)
            obs = entity_tracker.update(frame.frame_id, tracked)
            all_observations.append(obs)

        # Frames 0..2: both objects VISIBLE (entity 0 and entity 1)
        for frame_id in range(3):
            obs_map = {o.entity_id: o for o in all_observations[frame_id]}
            assert 0 in obs_map and obs_map[0].state == EntityState.VISIBLE
            assert 1 in obs_map and obs_map[1].state == EntityState.VISIBLE

        # Frame 3 (gap 1): entity 1 OCCLUDED
        obs_map_3 = {o.entity_id: o for o in all_observations[3]}
        assert obs_map_3[1].state == EntityState.OCCLUDED

        # Frame 4 (gap 2): entity 1 PREDICTED
        obs_map_4 = {o.entity_id: o for o in all_observations[4]}
        assert obs_map_4[1].state == EntityState.PREDICTED

        # Frame 5 (gap 3): entity 1 PREDICTED
        obs_map_5 = {o.entity_id: o for o in all_observations[5]}
        assert obs_map_5[1].state == EntityState.PREDICTED

        # Frame 6 (gap 4): entity 1 LOST
        obs_map_6 = {o.entity_id: o for o in all_observations[6]}
        assert obs_map_6[1].state == EntityState.LOST

        # Frame 7 (gap 5): entity 1 LOST
        obs_map_7 = {o.entity_id: o for o in all_observations[7]}
        assert obs_map_7[1].state == EntityState.LOST

        # Frame 8 (gap 6): entity 1 RETIRED and entered SpatialReidentifier pool
        obs_map_8 = {o.entity_id: o for o in all_observations[8]}
        assert obs_map_8[1].state == EntityState.RETIRED

        # Frame 9 (reappearance): Object 1 reappears in frame.
        # Upstream GreedyIoUTracker (max_age=1) issued a new track_id.
        # PersistentEntityTracker matches Candidate 1 from the retention pool and
        # re-links it to the original entity_id=1!
        obs_map_9 = {o.entity_id: o for o in all_observations[9]}
        assert 1 in obs_map_9
        assert obs_map_9[1].state == EntityState.VISIBLE
        # Verify a brand-new entity (e.g. entity_id=2) was NOT minted!
        assert 2 not in obs_map_9

