"""Tests for spatial re-identification and candidate retention pool (PR-007).

Covers ``ReidentificationCandidate`` creation and trajectory prediction,
``SpatialReidentifier`` threshold matching, rejection, retention window expiry,
and the mandatory hand-computed multi-candidate spatial disambiguation test.
Also pins ``SpatialReidentifier.match``'s exact behavior after the
``plausibility_score`` extraction (ADR-0011).
"""

import math

import pytest

from sentinel_vision.data.contracts import BoundingBox, Detection
from sentinel_vision.reidentification.spatial import (
    ReidentificationCandidate,
    SpatialReidentifier,
    plausibility_score,
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

    def test_predict_box_clamps_axes_that_would_invert(self) -> None:
        cand = ReidentificationCandidate(
            entity_id=1,
            last_known_box=BoundingBox(10.0, 10.0, 20.0, 20.0),
            velocity=(2.0, 1.0, -2.0, 1.0),
            retired_frame_id=5,
            last_observed_frame_id=5,
            class_label="synthetic_target",
        )
        # Raw projection at frame 14 (elapsed 9): x_min=28, x_max=2 (inverted).
        # The axis is re-centered on midpoint 15 with the last known width 10.
        pred_box = cand.predict_box(14)
        assert pred_box.x_min == 10.0
        assert pred_box.x_max == 20.0
        assert pred_box == BoundingBox(10.0, 19.0, 20.0, 29.0)


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


class TestPlausibilityScoreExtractionRegression:
    """ADR-0011 regression guard on the ``plausibility_score`` extraction.

    ``plausibility_score`` was lifted out of ``SpatialReidentifier.match`` so the
    retired pool and the active-entity pool score candidates with one function.
    This class pins ``match``'s observable behavior for a fixed set of inputs so
    the extraction cannot silently change a threshold, a comparison direction, or
    a tie-break: same inputs must produce the same outputs it produced before the
    refactor. Every expected value below is hand-computed from the documented
    rule (class equality, then ``dist > max_distance`` / ``overlap < min_iou``
    rejection, then lowest ``(dist, entity_id)``).
    """

    def _candidate(
        self,
        entity_id: int,
        box: BoundingBox,
        velocity: tuple[float, float, float, float],
        retired_frame_id: int,
        last_observed_frame_id: int,
        class_label: str = "synthetic_target",
    ) -> ReidentificationCandidate:
        return ReidentificationCandidate(
            entity_id=entity_id,
            last_known_box=box,
            velocity=velocity,
            retired_frame_id=retired_frame_id,
            last_observed_frame_id=last_observed_frame_id,
            class_label=class_label,
        )

    @pytest.mark.parametrize(
        "detection_box, detection_label, candidate_box, candidate_label, "
        "max_distance, min_iou, expected",
        [
            # Class match, no constraints configured: always plausible, and the
            # score is the raw center distance.
            (
                BoundingBox(12.0, 10.0, 22.0, 20.0),
                "synthetic_target",
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                None,
                None,
                math.hypot(2.0, 0.0),
            ),
            # Class mismatch: implausible regardless of geometry or thresholds.
            (
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "person",
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                50.0,
                None,
                None,
            ),
            # Distance exactly equal to max_distance stays plausible (strict >).
            (
                BoundingBox(15.0, 10.0, 25.0, 20.0),
                "synthetic_target",
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                5.0,
                None,
                math.hypot(5.0, 0.0),
            ),
            # One pixel past max_distance: rejected.
            (
                BoundingBox(15.0, 10.0, 25.0, 20.0),
                "synthetic_target",
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                4.9,
                None,
                None,
            ),
            # Identical boxes: IoU 1.0, meets any min_iou up to 1.0 exactly.
            (
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                None,
                1.0,
                0.0,
            ),
            # Half-overlap: IoU = 50/150 = 0.3333..., rejected at min_iou 0.5.
            (
                BoundingBox(15.0, 10.0, 25.0, 20.0),
                "synthetic_target",
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                None,
                0.5,
                None,
            ),
            # Same pair at min_iou 0.3: plausible, and the score ignores IoU.
            (
                BoundingBox(15.0, 10.0, 25.0, 20.0),
                "synthetic_target",
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                None,
                0.3,
                math.hypot(5.0, 0.0),
            ),
            # Both constraints configured: both must hold.
            (
                BoundingBox(12.0, 10.0, 22.0, 20.0),
                "synthetic_target",
                BoundingBox(10.0, 10.0, 20.0, 20.0),
                "synthetic_target",
                50.0,
                0.3,
                math.hypot(2.0, 0.0),
            ),
        ],
    )
    def test_plausibility_score_matches_the_pre_extraction_rule(
        self,
        detection_box: BoundingBox,
        detection_label: str,
        candidate_box: BoundingBox,
        candidate_label: str,
        max_distance: float | None,
        min_iou: float | None,
        expected: float | None,
    ) -> None:
        assert (
            plausibility_score(
                detection_box,
                detection_label,
                candidate_box,
                candidate_label,
                max_distance,
                min_iou,
            )
            == expected
        )

    def test_match_behavior_is_unchanged_for_a_fixed_input_table(self) -> None:
        """``match`` must return exactly what it returned before the extraction."""
        detection = Detection(
            bounding_box=BoundingBox(21.0, 11.0, 31.0, 21.0),
            confidence=1.0,
            class_label="synthetic_target",
        )
        # Candidate A: retired at frame 5, observed at frame 5, moving +2/frame.
        # At frame 10 its predicted box is (20,10,30,20), center (25,15);
        # detection center is (26,16) -> prediction error hypot(1,1).
        cand_a = self._candidate(
            10, BoundingBox(10.0, 10.0, 20.0, 20.0), (2.0, 0.0, 2.0, 0.0), 5, 5
        )
        # Candidate B: static at (100,100,110,110), center (105,105);
        # distance from the detection center is hypot(79, 89).
        cand_b = self._candidate(
            20, BoundingBox(100.0, 100.0, 110.0, 110.0), (0.0, 0.0, 0.0, 0.0), 8, 8
        )

        # max_distance 2.0: A (error 1.414) matches, B is out of range.
        tight = SpatialReidentifier(retention_window=20, max_distance=2.0)
        tight.add_candidate(cand_a)
        tight.add_candidate(cand_b)
        matched = tight.match(detection, frame_id=10)
        assert matched is not None
        assert matched is cand_a
        # The winner is removed from the pool; the rejected one is not.
        assert [c.entity_id for c in tight.candidates] == [20]

        # max_distance 150.0: both plausible, closest error wins, and the score
        # match reports is exactly the score match ranked on.
        loose = SpatialReidentifier(retention_window=20, max_distance=150.0)
        loose.add_candidate(cand_a)
        loose.add_candidate(cand_b)
        assert (
            plausibility_score(
                detection.bounding_box,
                detection.class_label,
                cand_a.predict_box(10),
                cand_a.class_label,
                150.0,
                None,
            )
            == math.hypot(1.0, 1.0)
        )
        assert (
            plausibility_score(
                detection.bounding_box,
                detection.class_label,
                cand_b.predict_box(10),
                cand_b.class_label,
                150.0,
                None,
            )
            == math.hypot(79.0, 89.0)
        )
        matched = loose.match(detection, frame_id=10)
        assert matched is not None
        assert matched is cand_a
        assert [c.entity_id for c in loose.candidates] == [20]

        # No candidate within max_distance at all: None, pool untouched.
        none_match = SpatialReidentifier(retention_window=20, max_distance=1.0)
        none_match.add_candidate(cand_a)
        none_match.add_candidate(cand_b)
        assert none_match.match(detection, frame_id=10) is None
        assert [c.entity_id for c in none_match.candidates] == [10, 20]

        # Class mismatch: None even at zero distance, pool untouched.
        other_class = SpatialReidentifier(retention_window=20, max_distance=150.0)
        other_class.add_candidate(
            self._candidate(
                30,
                BoundingBox(21.0, 11.0, 31.0, 21.0),
                (0.0, 0.0, 0.0, 0.0),
                10,
                10,
                class_label="forklift",
            )
        )
        assert other_class.match(detection, frame_id=10) is None
        assert [c.entity_id for c in other_class.candidates] == [30]

        # min_iou-only configuration keeps working the same way.
        iou_only = SpatialReidentifier(retention_window=20, max_distance=None, min_iou=0.2)
        iou_only.add_candidate(
            self._candidate(
                40, BoundingBox(21.0, 11.0, 31.0, 21.0), (0.0, 0.0, 0.0, 0.0), 10, 10
            )
        )
        matched = iou_only.match(detection, frame_id=10)
        assert matched is not None
        assert matched.entity_id == 40
        assert len(iou_only.candidates) == 0

    def test_configured_thresholds_are_readable(self) -> None:
        """The active-entity pool reads the same thresholds match uses."""
        reidentifier = SpatialReidentifier(
            retention_window=7, max_distance=12.5, min_iou=0.25
        )
        assert reidentifier.retention_window == 7
        assert reidentifier.max_distance == 12.5
        assert reidentifier.min_iou == 0.25

        unset = SpatialReidentifier(retention_window=3, max_distance=None, min_iou=0.5)
        assert unset.max_distance is None
        assert unset.min_iou == 0.5
