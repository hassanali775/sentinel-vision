"""Integration test for the full deterministic event pipeline (PR-009).

Full pipeline: MultiObjectSyntheticFrameStream (two objects) ->
SyntheticBoxDetector(multi_instance=True) -> GreedyIoUTracker ->
PersistentEntityTracker -> WorkspaceModel.evaluate() -> EventEngine.

Object 0 (entity 0) travels right along lane y=15 with detected box center
(15 + 2*frame, 15); object 1 (entity 1) travels left along lane y=35 with
center (85 - 3*frame, 35). Their center distance is sqrt((5*f - 70)^2 + 20^2),
which drops strictly below 25 px from frame 12 to frame 16. Entity 0's center
is inside zone lane_a (x in [40, 60], y in [0, 25]) from frame 13 to frame 22.

The test asserts the proximity hazard and zone intrusion events open and close
at exactly those hand-computed frames, pinning the whole deterministic core.
"""

from sentinel_vision.detection.synthetic import SyntheticBoxDetector
from sentinel_vision.events.engine import EventEngine
from sentinel_vision.events.event import EventStatus, EventType
from sentinel_vision.events.rules import ProximityHazardRule, ZoneIntrusionRule
from sentinel_vision.ingestion.stream import (
    MultiObjectSyntheticFrameStream,
    SyntheticObjectConfig,
)
from sentinel_vision.spatial.workspace import WorkspaceModel
from sentinel_vision.spatial.zone import Zone
from sentinel_vision.state.tracker import PersistentEntityTracker
from sentinel_vision.tracking.greedy import GreedyIoUTracker


class TestEventPipeline:
    def test_proximity_and_zone_events_at_hand_computed_frames(self) -> None:
        obj0 = SyntheticObjectConfig(
            start_x=10.0,
            start_y=10.0,
            velocity_x=2.0,
            velocity_y=0.0,
            width=10,
            height=10,
        )
        obj1 = SyntheticObjectConfig(
            start_x=80.0,
            start_y=30.0,
            velocity_x=-3.0,
            velocity_y=0.0,
            width=10,
            height=10,
        )
        stream = MultiObjectSyntheticFrameStream(
            width=200, height=100, fps=10.0, num_frames=25, objects=[obj0, obj1]
        )

        detector = SyntheticBoxDetector(multi_instance=True)
        tracker = GreedyIoUTracker(iou_threshold=0.5, max_age=10)
        entity_tracker = PersistentEntityTracker(
            occlusion_budget=1, prediction_budget=3, retirement_budget=5
        )
        lane_a = Zone(
            name="lane_a",
            vertices=[(40.0, 0.0), (60.0, 0.0), (60.0, 25.0), (40.0, 25.0)],
        )
        model = WorkspaceModel(zones=[lane_a])
        engine = EventEngine(
            rules=[
                (ProximityHazardRule(threshold_px=25.0), 1, 1),
                (ZoneIntrusionRule(zone_name="lane_a"), 1, 1),
            ]
        )

        events_by_frame: dict[int, list] = {}
        for frame in stream:
            detections = detector.detect(frame)
            tracked = tracker.track(frame, detections)
            observations = entity_tracker.update(frame.frame_id, tracked)
            spatial = model.evaluate(frame.frame_id, observations)
            events_by_frame[frame.frame_id] = engine.update(
                frame.frame_id, spatial
            )

        # Object 0 is the first detection (sorted by y_min) and becomes
        # entity 0; object 1 becomes entity 1.
        for frame_id in range(12):
            assert events_by_frame[frame_id] == []

        # Frame 12: centers (39, 15) and (49, 35) are sqrt(500) ~= 22.4 px
        # apart, below the 25 px threshold -> proximity hazard opens.
        prox_open = events_by_frame[12]
        assert len(prox_open) == 1
        assert prox_open[0].event_type is EventType.PROXIMITY_HAZARD
        assert prox_open[0].entity_ids == (0, 1)
        assert prox_open[0].status is EventStatus.OPEN
        assert prox_open[0].opened_frame_id == 12
        assert prox_open[0].zone_name is None

        # Frame 13: entity 0's center (41, 15) has crossed into lane_a.
        zone_open = events_by_frame[13]
        assert len(zone_open) == 1
        assert zone_open[0].event_type is EventType.ZONE_INTRUSION
        assert zone_open[0].entity_ids == (0,)
        assert zone_open[0].status is EventStatus.OPEN
        assert zone_open[0].opened_frame_id == 13
        assert zone_open[0].zone_name == "lane_a"

        for frame_id in range(14, 17):
            assert events_by_frame[frame_id] == []

        # Frame 17: centers (49, 15) and (34, 35) are exactly 25 px apart,
        # no longer strictly below the threshold -> proximity hazard closes.
        prox_close = events_by_frame[17]
        assert len(prox_close) == 1
        assert prox_close[0].event_type is EventType.PROXIMITY_HAZARD
        assert prox_close[0].entity_ids == (0, 1)
        assert prox_close[0].status is EventStatus.CLOSED
        assert prox_close[0].opened_frame_id == 12
        assert prox_close[0].closed_frame_id == 17

        for frame_id in range(18, 23):
            assert events_by_frame[frame_id] == []

        # Frame 23: entity 0's center (61, 15) has crossed out of lane_a.
        zone_close = events_by_frame[23]
        assert len(zone_close) == 1
        assert zone_close[0].event_type is EventType.ZONE_INTRUSION
        assert zone_close[0].entity_ids == (0,)
        assert zone_close[0].status is EventStatus.CLOSED
        assert zone_close[0].opened_frame_id == 13
        assert zone_close[0].closed_frame_id == 23
