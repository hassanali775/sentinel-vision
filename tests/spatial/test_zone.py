"""Tests for Zone point-in-polygon geometry (PR-008).

Covers the ray-casting point-in-polygon test (clearly inside, clearly
outside, the documented half-open boundary behavior), a non-axis-aligned
polygon proving the test is real ray casting and not an axis-aligned box
check, and constructor rejection of degenerate polygons and empty names.
"""

import pytest

from sentinel_vision.spatial.zone import Zone

UNIT_SQUARE_VERTICES = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def unit_square(name: str = "unit_square") -> Zone:
    return Zone(name=name, vertices=UNIT_SQUARE_VERTICES)


class TestZone:
    def test_zone_construction_and_properties(self) -> None:
        zone = unit_square("workspace_a")
        assert zone.name == "workspace_a"
        assert zone.vertices == UNIT_SQUARE_VERTICES

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Zone(name="", vertices=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        with pytest.raises(ValueError, match="name"):
            Zone(name="   ", vertices=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])

    def test_rejects_less_than_three_vertices(self) -> None:
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Zone(name="degenerate", vertices=[(0.0, 0.0), (10.0, 0.0)])
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Zone(name="degenerate", vertices=[(0.0, 0.0)])

    def test_point_clearly_inside(self) -> None:
        zone = unit_square()
        assert zone.contains_point(5.0, 5.0) is True

    def test_point_clearly_outside(self) -> None:
        zone = unit_square()
        assert zone.contains_point(15.0, 15.0) is False
        assert zone.contains_point(-5.0, 5.0) is False
        assert zone.contains_point(5.0, 25.0) is False

    def test_boundary_points_resolve_via_half_open_convention(self) -> None:
        """Pins the documented on-boundary behavior of the ray-casting rule.

        The even-odd test is half-open, so it has no explicit on-edge rule.
        For the unit square this implementation reports the left and bottom
        edges as inside and the right and top edges as outside.
        """
        zone = unit_square()
        assert zone.contains_point(0.0, 5.0) is True
        assert zone.contains_point(5.0, 0.0) is True
        assert zone.contains_point(10.0, 5.0) is False
        assert zone.contains_point(5.0, 10.0) is False

    def test_triangle_uses_ray_casting_not_axis_aligned_box(self) -> None:
        # Triangle with base y=0 from x=0..10 and apex (5,10). The right edge
        # runs from (10,0) to (5,10): at y=3 it crosses x=8.5, at y=1 it
        # crosses x=9.5. A point like (9,3) is inside the AABB but outside the
        # triangle, proving the test is a real polygon test.
        triangle = Zone(
            name="triangle", vertices=[(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        )
        assert triangle.contains_point(5.0, 3.0) is True
        assert triangle.contains_point(9.0, 3.0) is False
        assert triangle.contains_point(5.0, 12.0) is False
