"""Tests for spatial re-identification and candidate retention pool (PR-007).

Covers ``ReidentificationCandidate`` creation and trajectory prediction,
``SpatialReidentifier`` threshold matching, rejection, retention window expiry,
and the mandatory hand-computed multi-candidate spatial disambiguation test.
"""

import pytest

from sentinel_vision.data.contracts import BoundingBox, Detection
from sentinel_vision.reidentification.spatial import (
    ReidentificationCandidate,
    SpatialReidentifier,
)


class TestReidentificationCandidate:
    def test_candidate_creation_and_properties(self) -> None:
        cand = ReidentificationCandidate(
            entity_id=5,
            last_known_box=BoundingBox(10.0, 20.0, 30.0, 40.0),
            velocity=(1.0, 0.5, 1.0, 0.5),
            retired_frame_id=10,
            last_observed_frame_id=7,
            class_label="synthetic_target",
        )
        assert cand.entity_id == 5
        assert cand.last_known_box == BoundingBox(10.0, 20.0, 30.0, 40.0)
        assert cand.velocity == (1.0, 0.5, 1.0, 0.5)
        assert cand.retired_frame_id == 10
        assert cand.last_observed_frame_id == 7
        assert cand.class_label == "synthetic_target"

    def test_rejects_negative_entity_id(self) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            ReidentificationCandidate(
                entity_id=-1,
                last_known_box=BoundingBox(0, 0, 10, 10),
                velocity=(0, 0, 0, 0),
                retired_frame_id=5,
                last_observed_frame_id=2,
                class_label="synthetic_target",
            )

    def test_rejects_negative_retired_frame_id(self) -> None:
        with pytest.raises(ValueError, match="retired_frame_id"):
            ReidentificationCandidate(
                entity_id=0,
                last_known_box=BoundingBox(0, 0, 10, 10),
                velocity=(0, 0, 0, 0),
                retired_frame_id=-1,
                last_observed_frame_id=0,
                class_label="synthetic_target",
            )

    def test_rejects_negative_last_observed_frame_id(self) -> None:
        with pytest.raises(ValueError, match="last_observed_frame_id"):
            ReidentificationCandidate(
                entity_id=0,
                last_known_box=BoundingBox(0, 0, 10, 10),
                velocity=(0, 0, 0, 0),
                retired_frame_id=5,
                last_observed_frame_id=-1,
                class_label="synthetic_target",
            )

    def test_rejects_last_observed_after_retired_frame_id(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            ReidentificationCandidate(
                entity_id=0,
                last_known_box=BoundingBox(0, 0, 10, 10),
                velocity=(0, 0, 0, 0),
                retired_frame_id=5,
                last_observed_frame_id=6,
                class_label="synthetic_target",
            )

    def test_predict_box_extrapolation(self) -> None:
        cand = ReidentificationCandidate(
            entity_id=1,
            last_known_box=BoundingBox(10.0, 10.0, 20.0, 20.0),
            velocity=(2.0, 1.0, 2.0, 1.0),
            retired_frame_id=5,
            last_observed_frame_id=5,
            class_label="synthetic_target",
        )
        # At frame 8 (3 frames after last_observed_frame_id 5):
        # x_min = 10 + 3*2 = 16, y_min = 10 + 3*1 = 13
        # x_max = 20 + 3*2 = 26, y_max = 20 + 3*1 = 23
        pred_box = cand.predict_box(8)
        assert pred_box == BoundingBox(16.0, 13.0, 26.0, 23.0)

    def test_predict_box_rejects_frame_id_before_retired(self) -> None:
        cand = ReidentificationCandidate(
            entity_id=1,
            last_known_box=BoundingBox(10, 10, 20, 20),
            velocity=(0, 0, 0, 0),
            retired_frame_id=5,
            last_observed_frame_id=3,
            class_label="synthetic_target",
        )
        with pytest.raises(ValueError, match="cannot be before"):
            cand.predict_box(4)


class TestSpatialReidentifier:
    def test_constructor_parameter_validation(self) -> None:
        with pytest.raises(ValueError, match="retention_window"):
            SpatialReidentifier(retention_window=-1)

        with pytest.raises(ValueError, match="At least one"):
            SpatialReidentifier(max_distance=None, min_iou=None)

        with pytest.raises(ValueError, match="max_distance"):
            SpatialReidentifier(max_distance=-5.0)

        with pytest.raises(ValueError, match="min_iou"):
            SpatialReidentifier(min_iou=1.5)

    def test_successful_reid_within_threshold(self) -> None:
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=15.0)
        cand = ReidentificationCandidate(
            entity_id=42,
            last_known_box=BoundingBox(10.0, 10.0, 20.0, 20.0),
            velocity=(1.0, 0.0, 1.0, 0.0),
            retired_frame_id=5,
            last_observed_frame_id=5,
            class_label="synthetic_target",
        )
        reidentifier.add_candidate(cand)
        assert len(reidentifier.candidates) == 1

        # At frame 7 (2 frames later), predicted box is (12.0, 10.0, 22.0, 20.0),
        # center (17.0, 15.0)
        detection = Detection(
            bounding_box=BoundingBox(12.5, 10.5, 22.5, 20.5),  # center (17.5, 15.5), dist ~ 0.707
            confidence=1.0,
            class_label="synthetic_target",
        )
        matched = reidentifier.match(detection, frame_id=7)
        assert matched is not None
        assert matched.entity_id == 42
        # Candidate removed from retention pool on successful match
        assert len(reidentifier.candidates) == 0

    def test_rejection_outside_threshold(self) -> None:
        reidentifier = SpatialReidentifier(retention_window=10, max_distance=5.0)
        cand = ReidentificationCandidate(
            entity_id=7,
            last_known_box=BoundingBox(10.0, 10.0, 20.0, 20.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            retired_frame_id=5,
            last_observed_frame_id=5,
            class_label="synthetic_target",
        )
        reidentifier.add_candidate(cand)

        # Detection far away (center 100, 100 vs predicted center 15, 15)
        detection = Detection(
            bounding_box=BoundingBox(95.0, 95.0, 105.0, 105.0),
            confidence=1.0,
            class_label="synthetic_target",
        )
        matched = reidentifier.match(detection, frame_id=7)
        assert matched is None
        # Candidate remains in pool since it was not matched
        assert len(reidentifier.candidates) == 1

    def test_retention_window_expiry(self) -> None:
        reidentifier = SpatialReidentifier(retention_window=5, max_distance=50.0)
        cand = ReidentificationCandidate(
            entity_id=99,
            last_known_box=BoundingBox(10.0, 10.0, 20.0, 20.0),
            velocity=(0.0, 0.0, 0.0, 0.0),
            retired_frame_id=10,
            last_observed_frame_id=10,
            class_label="synthetic_target",
        )
        reidentifier.add_candidate(cand)

        # Frame 15: age = 15 - 10 = 5 <= retention_window (5) -> still valid
        reidentifier.purge_expired(15)
        assert len(reidentifier.candidates) == 1

        # Frame 16: age = 16 - 10 = 6 > retention_window (5) -> permanently purged
        detection = Detection(
            bounding_box=BoundingBox(10.0, 10.0, 20.0, 20.0),
            confidence=1.0,
            class_label="synthetic_target",
        )
        matched = reidentifier.match(detection, frame_id=16)
        assert matched is None
        assert len(reidentifier.candidates) == 0  # Candidate permanently purged

    def test_multi_candidate_disambiguation_closest_prediction(self) -> None:
        """Mandatory Disambiguation Test:

        Proves that when multiple retired candidates are spatially plausible, the match
        goes strictly to the candidate with the smallest prediction error (closest to
        its predicted trajectory) regardless of candidate pool order or retirement recency.
        """
        # Hand-computed scenario:
        # Candidate A (entity_id=10):
        #   retired at frame 5, last observed at frame 5
        #   last_known_box = (10, 10, 20, 20)
        #   velocity = (2.0, 0.0, 2.0, 0.0)
        #   At frame 10 (5 elapsed frames): predicted box = (20, 10, 30, 20), center = (25.0, 15.0)
        cand_a = ReidentificationCandidate(
            entity_id=10,
            last_known_box=BoundingBox(10.0, 10.0, 20.0, 20.0),
            velocity=(2.0, 0.0, 2.0, 0.0),
            retired_frame_id=5,
            last_observed_frame_id=5,
            class_label="synthetic_target",
        )

        # Candidate B (entity_id=20):
        #   retired at frame 8 (more recently than A!), last observed at frame 8
        #   last_known_box = (100, 100, 110, 110)
        #   velocity = (0.0, 3.0, 0.0, 3.0)
        #   At frame 10 (2 elapsed frames): predicted box = (100, 106, 110, 116),
        #   center = (105.0, 111.0)
        cand_b = ReidentificationCandidate(
            entity_id=20,
            last_known_box=BoundingBox(100.0, 100.0, 110.0, 110.0),
            velocity=(0.0, 3.0, 0.0, 3.0),
            retired_frame_id=8,
            last_observed_frame_id=8,
            class_label="synthetic_target",
        )

        # Detection D at frame 10: box = (21, 11, 31, 21), center = (26.0, 16.0)
        # Distance to A's predicted center (25.0, 15.0) = sqrt(1^2 + 1^2) = ~1.414 px
        # Distance to B's predicted center (105.0, 111.0) = sqrt(79^2 + 95^2) = ~123.556 px
        detection = Detection(
            bounding_box=BoundingBox(21.0, 11.0, 31.0, 21.0),
            confidence=1.0,
            class_label="synthetic_target",
        )

        # Test Case 1: Pool order [Candidate A, Candidate B]
        pool1 = SpatialReidentifier(retention_window=20, max_distance=150.0)
        pool1.add_candidate(cand_a)
        pool1.add_candidate(cand_b)
        match1 = pool1.match(detection, frame_id=10)
        assert match1 is not None
        assert match1.entity_id == 10  # Selected Candidate A

        # Test Case 2: Swapped pool order [Candidate B, Candidate A], where B is
        # also retired more recently
        pool2 = SpatialReidentifier(retention_window=20, max_distance=150.0)
        pool2.add_candidate(cand_b)
        pool2.add_candidate(cand_a)
        match2 = pool2.match(detection, frame_id=10)
        assert match2 is not None
        assert match2.entity_id == 10  # MUST STILL select Candidate A specifically!
