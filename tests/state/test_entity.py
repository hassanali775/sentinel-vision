"""Tests for entity state contracts and invariants (PR-006).

Covers EntityObservation validation: non-negative IDs, and the
LOST-iff-None-box invariant (both ways).
"""

import pytest

from sentinel_vision.data.contracts import BoundingBox
from sentinel_vision.state.entity import EntityObservation, EntityState


def sample_box() -> BoundingBox:
    return BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)


class TestEntityObservation:
    def test_valid_entity_observation_construction(self) -> None:
        box = sample_box()
        obs_visible = EntityObservation(
            entity_id=0,
            state=EntityState.VISIBLE,
            bounding_box=box,
            class_label="target",
            frame_id=0,
        )
        assert obs_visible.entity_id == 0
        assert obs_visible.state == EntityState.VISIBLE
        assert obs_visible.bounding_box == box

        obs_lost = EntityObservation(
            entity_id=1,
            state=EntityState.LOST,
            bounding_box=None,
            class_label="target",
            frame_id=1,
        )
        assert obs_lost.state == EntityState.LOST
        assert obs_lost.bounding_box is None

    def test_lost_state_with_box_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="bounding_box must be None when state is LOST"):
            EntityObservation(
                entity_id=0,
                state=EntityState.LOST,
                bounding_box=sample_box(),
                class_label="target",
                frame_id=0,
            )

    @pytest.mark.parametrize(
        "state",
        [
            EntityState.VISIBLE,
            EntityState.OCCLUDED,
            EntityState.PREDICTED,
            EntityState.RETIRED,
        ],
    )
    def test_non_lost_state_with_none_box_raises_value_error(
        self, state: EntityState
    ) -> None:
        with pytest.raises(
            ValueError, match="bounding_box cannot be None when state is"
        ):
            EntityObservation(
                entity_id=0,
                state=state,
                bounding_box=None,
                class_label="target",
                frame_id=0,
            )

    def test_rejects_negative_entity_id(self) -> None:
        with pytest.raises(ValueError, match="entity_id"):
            EntityObservation(
                entity_id=-1,
                state=EntityState.VISIBLE,
                bounding_box=sample_box(),
                class_label="target",
                frame_id=0,
            )

    def test_rejects_negative_frame_id(self) -> None:
        with pytest.raises(ValueError, match="frame_id"):
            EntityObservation(
                entity_id=0,
                state=EntityState.VISIBLE,
                bounding_box=sample_box(),
                class_label="target",
                frame_id=-1,
            )
