"""Pure-arithmetic box geometry for detection evaluation."""

from __future__ import annotations

from sentinel_vision.data.contracts import BoundingBox


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two boxes, in ``[0.0, 1.0]``.

    Returns ``0.0`` for boxes that do not overlap (including boxes that only
    touch at an edge or corner) — non-overlap is a valid, common case, not
    an error. Because boxes are validated to have positive width and height,
    the union in the denominator is always positive when the intersection is.
    """

    inter_x_min = max(a.x_min, b.x_min)
    inter_y_min = max(a.y_min, b.y_min)
    inter_x_max = min(a.x_max, b.x_max)
    inter_y_max = min(a.y_max, b.y_max)

    inter_width = inter_x_max - inter_x_min
    inter_height = inter_y_max - inter_y_min
    if inter_width <= 0.0 or inter_height <= 0.0:
        return 0.0

    intersection = inter_width * inter_height
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = area_a + area_b - intersection
    return intersection / union
