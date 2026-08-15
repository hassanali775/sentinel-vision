"""Integration test for the full spatial workspace pipeline (PR-008).

Full pipeline: MultiObjectSyntheticFrameStream (two objects) ->
SyntheticBoxDetector(multi_instance=True) -> GreedyIoUTracker ->
PersistentEntityTracker -> WorkspaceModel.evaluate().

Object 0 travels along lane y=15 with box center_x = 15 + 2*frame; object 1
travels along lane y=55. The zone covers x in [30, 70] on y in [0, 40]. The
test asserts zone membership turning on/off at hand-computed crossing frames
and a hand-computed pairwise distance.
"""

import pytest

from sentinel_vision.detection.synthetic import SyntheticBoxDetector
from sentinel_vision.ingestion.stream import (
    MultiObjectSyntheticFrameStream,
    SyntheticObjectConfig,
)
from sentinel_vision.spatial.workspace import WorkspaceModel
from sentinel_vision.spatial.zone import Zone
from sentinel_vision.state.entity import EntityObservation
from sentinel_vision.state.tracker import PersistentEntityTracker
from sentinel_vision.tracking.greedy import GreedyIoUTracker


class TestSpatialPipeline:
    def test_zone_membership_turns_on_and_off_at_hand_computed_frames(self) -> None:
        obj0 = SyntheticObjectConfig(
            start_x=10.0,
            start_y=10.0,
            velocity_x=2.0,
            velocity_y=0.0,
            width=10,
            height=10,
        )
        obj1 = SyntheticObjectConfig(
            start_x=10.0,
            start_y=50.0,
            velocity_x=2.0,
            velocity_y=0.0,
            width=10,
            height=10,
        )
        stream = MultiObjectSyntheticFrameStream(
            width=200, height=100, fps=10.0, num_frames=40, objects=[obj0, obj1]
        )

        detector = SyntheticBoxDetector(multi_instance=True)
        tracker = GreedyIoUTracker(iou_threshold=0.5, max_age=10)
        entity_tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )

        all_observations: list[list[EntityObservation]] = []
        for frame in stream:
            detections = detector.detect(frame)
            tracked = tracker.track(frame, detections)
            all_observations.append(entity_tracker.update(frame.frame_id, tracked))

        # Object 0 is the first detection (sorted by y_min) and becomes
        # entity 0. Its detected box center is (15 + 2*frame, 15).
        # center_x crosses x=30 between frame 7 (29) and frame 8 (31), and
        # crosses x=70 between frame 27 (69) and frame 28 (71).
        lane_a = Zone(
            name="lane_a",
            vertices=[(30.0, 0.0), (70.0, 0.0), (70.0, 40.0), (30.0, 40.0)],
        )
        model = WorkspaceModel(zones=[lane_a])

        # Frame 7: center (29, 15) is just west of the zone -> not a member.
        frame7 = model.evaluate(7, all_observations[7])
        assert frame7.zone_memberships[0] == []
        assert frame7.zone_memberships[1] == []

        # Frame 8: center (31, 15) has crossed into the zone -> member.
        frame8 = model.evaluate(8, all_observations[8])
        assert frame8.zone_memberships[0] == ["lane_a"]
        assert frame8.zone_memberships[1] == []

        # Frame 27: center (69, 15) is still inside (west of x=70).
        frame27 = model.evaluate(27, all_observations[27])
        assert frame27.zone_memberships[0] == ["lane_a"]

        # Frame 28: center (71, 15) has crossed out -> not a member again.
        frame28 = model.evaluate(28, all_observations[28])
        assert frame28.zone_memberships[0] == []
        assert frame28.zone_memberships[1] == []

        # Hand-computed pairwise distance at every frame: object centers sit at
        # (15 + 2*frame, 15) and (15 + 2*frame, 55), exactly 40 px apart.
        assert frame8.pairwise_distances[(0, 1)] == pytest.approx(40.0)
