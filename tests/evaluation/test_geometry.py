"""Tests for IoU geometry (PR-002, Phase 2).

Covers full overlap, no overlap, a hand-computed partial overlap, a
contained box, and edge/corner-touching boxes (which must yield 0.0, not a
division error).
"""

import pytest

from sentinel_vision.data.contracts import BoundingBox
from sentinel_vision.evaluation.geometry import iou


def box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


class TestIoU:
    def test_full_overlap_is_one(self) -> None:
        a = box(0.0, 0.0, 10.0, 10.0)
        assert iou(a, a) == 1.0

    def test_identical_area_different_position_no_overlap(self) -> None:
        a = box(0.0, 0.0, 10.0, 10.0)
        b = box(20.0, 20.0, 30.0, 30.0)
        assert iou(a, b) == 0.0

    def test_partial_overlap_hand_computed(self) -> None:
        # a: (0,0,10,10), b: (5,0,15,10) -> intersection 5x10=50,
        # union = 100 + 100 - 50 = 150, IoU = 50/150 = 1/3.
        a = box(0.0, 0.0, 10.0, 10.0)
        b = box(5.0, 0.0, 15.0, 10.0)
        assert iou(a, b) == pytest.approx(1 / 3)

    def test_contained_box_hand_computed(self) -> None:
        # b fully inside a: intersection = area(b) = 36,
        # union = area(a) = 100, IoU = 36/100.
        a = box(0.0, 0.0, 10.0, 10.0)
        b = box(2.0, 2.0, 8.0, 8.0)
        assert iou(a, b) == pytest.approx(0.36)

    def test_edge_touching_boxes_is_zero(self) -> None:
        a = box(0.0, 0.0, 10.0, 10.0)
        b = box(10.0, 0.0, 20.0, 10.0)
        assert iou(a, b) == 0.0

    def test_corner_touching_boxes_is_zero(self) -> None:
        a = box(0.0, 0.0, 10.0, 10.0)
        b = box(10.0, 10.0, 20.0, 20.0)
        assert iou(a, b) == 0.0

    def test_iou_is_symmetric(self) -> None:
        a = box(0.0, 0.0, 10.0, 10.0)
        b = box(5.0, 0.0, 15.0, 10.0)
        assert iou(a, b) == iou(b, a)
