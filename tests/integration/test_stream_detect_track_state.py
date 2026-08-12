"""Integration tests for Stream -> Detect -> Track -> PersistentEntityTracker pipeline (PR-006).

Tests full pipeline state progression (VISIBLE -> OCCLUDED -> PREDICTED -> VISIBLE / RETIRED)
and proves the tracker-composition limitation when upstream max_age is shorter
than entity budget.
"""

from sentinel_vision.detection.synthetic import SyntheticBoxDetector
from sentinel_vision.ingestion.stream import SyntheticFrameStream
from sentinel_vision.state.entity import EntityObservation, EntityState
from sentinel_vision.state.tracker import PersistentEntityTracker
from sentinel_vision.tracking.greedy import GreedyIoUTracker


class TestStreamDetectTrackState:
    def test_full_pipeline_entity_lifecycle_with_generous_max_age(self) -> None:
        stream = SyntheticFrameStream(width=128, height=96, fps=10.0, num_frames=150)
        detector = SyntheticBoxDetector()

        # Upstream tracker max_age=10 is >= PersistentEntityTracker total budget (5)
        tracker = GreedyIoUTracker(iou_threshold=0.5, max_age=10)
        entity_tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )

        frames = [next(stream) for _ in range(10)]
        all_observations: list[list[EntityObservation]] = []

        # Manufactured gap: suppress detections on frames 2, 3, 4
        for index, frame in enumerate(frames):
            detections = [] if index in (2, 3, 4) else detector.detect(frame)
            tracked = tracker.track(frame, detections)
            obs = entity_tracker.update(frame.frame_id, tracked)
            all_observations.append(obs)

        # Frame 0: VISIBLE (entity 0)
        assert all_observations[0][0].entity_id == 0
        assert all_observations[0][0].state == EntityState.VISIBLE

        # Frame 1: VISIBLE (entity 0)
        assert all_observations[1][0].entity_id == 0
        assert all_observations[1][0].state == EntityState.VISIBLE

        # Frame 2 (gap frame 1, k=1 <= 1): OCCLUDED
        assert all_observations[2][0].entity_id == 0
        assert all_observations[2][0].state == EntityState.OCCLUDED

        # Frame 3 (gap frame 2, k=2 <= 3): PREDICTED
        assert all_observations[3][0].entity_id == 0
        assert all_observations[3][0].state == EntityState.PREDICTED

        # Frame 4 (gap frame 3, k=3 <= 3): PREDICTED
        assert all_observations[4][0].entity_id == 0
        assert all_observations[4][0].state == EntityState.PREDICTED

        # Frame 5 (reappearance): VISIBLE (entity 0 preserved because max_age=10 kept track_id=0)
        assert all_observations[5][0].entity_id == 0
        assert all_observations[5][0].state == EntityState.VISIBLE

    def test_tracker_composition_limitation_with_short_max_age(self) -> None:
        """Proves the limitation documented in ADR-0006:

        When GreedyIoUTracker's max_age (1) is shorter than PersistentEntityTracker's
        total budget (5), a 2-frame gap drops track_id=0 upstream and reissues track_id=1.
        PersistentEntityTracker sees track_id=1 as a new object, so original entity 0
        proceeds toward LOST/RETIRED while a new entity 1 is created for track_id=1.
        """
        stream = SyntheticFrameStream(width=128, height=96, fps=10.0, num_frames=40)
        detector = SyntheticBoxDetector()

        # Upstream tracker max_age=1 is SHORTER than total entity budget (5)
        tracker = GreedyIoUTracker(iou_threshold=0.5, max_age=1)
        entity_tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )

        frames = [next(stream) for _ in range(8)]
        all_observations: list[list[EntityObservation]] = []

        # Gap on frames 2 and 3 (2 consecutive frames without detection)
        for index, frame in enumerate(frames):
            detections = [] if index in (2, 3) else detector.detect(frame)
            tracked = tracker.track(frame, detections)
            obs = entity_tracker.update(frame.frame_id, tracked)
            all_observations.append(obs)

        # Frame 0 & 1: entity 0 VISIBLE
        assert all_observations[0][0].entity_id == 0
        assert all_observations[0][0].state == EntityState.VISIBLE

        # Frame 2 (gap 1): entity 0 OCCLUDED
        assert all_observations[2][0].entity_id == 0
        assert all_observations[2][0].state == EntityState.OCCLUDED

        # Frame 3 (gap 2): entity 0 PREDICTED
        assert all_observations[3][0].entity_id == 0
        assert all_observations[3][0].state == EntityState.PREDICTED

        # Frame 4 (reappearance in frame stream): GreedyIoUTracker max_age=1 dropped track_id=0
        # and issued new track_id=1.
        # PersistentEntityTracker sees track_id=1 as UNSEEN -> creates Entity 1!
        # Entity 0 receives no match on frame 4 (k=3 <= 3) -> remains PREDICTED!
        frame4_obs = {obs.entity_id: obs for obs in all_observations[4]}
        assert 0 in frame4_obs
        assert frame4_obs[0].state == EntityState.PREDICTED
        assert 1 in frame4_obs
        assert frame4_obs[1].state == EntityState.VISIBLE

        # Frame 5: Entity 0 receives no match (k=4 <= 5) -> LOST. Entity 1 matched -> VISIBLE.
        frame5_obs = {obs.entity_id: obs for obs in all_observations[5]}
        assert frame5_obs[0].state == EntityState.LOST
        assert frame5_obs[1].state == EntityState.VISIBLE
