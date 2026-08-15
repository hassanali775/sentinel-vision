"""Zone geometry for the spatial workspace model (PR-008).

``Zone`` is a named polygon in image-pixel coordinates with a pure-Python
ray-casting point-in-polygon test. It deliberately avoids geometry
dependencies (shapely, scipy.spatial, OpenCV): the pipeline's dependency
ceiling is NumPy plus the standard library (ADR-0002, ADR-0003), and a
ray-casting point-in-polygon test is a handful of lines (see
docs/adr/0008-spatial-workspace-model.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    """A named pixel-space polygon used for spatial workspace facts.

    Attributes:
        name: Human-readable zone identifier. Must be a non-empty string
            after stripping whitespace.
        vertices: Polygon vertices in image-pixel coordinates as ``(x, y)``
            pairs, listed in boundary order (clockwise or counter-clockwise;
            either works for ray casting). Must contain at least three
            points.

    A degenerate polygon (fewer than three vertices) is a data-quality bug
    and is rejected in ``__post_init__``, following the project's contract
    validation convention.
    """

    name: str
    vertices: list[tuple[float, float]]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("zone name must be a non-empty string")
        if len(self.vertices) < 3:
            raise ValueError(
                f"zone '{self.name}' must have at least 3 vertices, "
                f"got {len(self.vertices)}"
            )
        object.__setattr__(self, "vertices", list(self.vertices))

    def contains_point(self, x: float, y: float) -> bool:
        """Return whether ``(x, y)`` lies inside this zone.

        Uses the classic even-odd ray-casting rule: a horizontal ray from
        ``(x, y)`` toward +x counts how many polygon edges it strictly
        crosses; an odd count means inside, an even count means outside.

        Boundary behavior: this implementation does not special-case points
        exactly on the polygon boundary. The crossing test is half-open — an
        edge straddles the ray only when one endpoint is strictly above the
        point's y-coordinate and the other is at or below it — so an
        on-boundary point resolves deterministically to whichever parity the
        count happens to produce, and callers must not rely on a consistent
        on-boundary answer. For the axis-aligned unit square used in the
        tests (vertices (0,0), (10,0), (10,10), (0,10)) this resolves as:
        points exactly on the left or bottom edge report inside (True), and
        points exactly on the right or top edge report outside (False). Both
        outcomes are pinned by tests in tests/spatial/test_zone.py.
        """
        inside = False
        j = len(self.vertices) - 1
        for i in range(len(self.vertices)):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            straddles = (yi > y) != (yj > y)
            if straddles:
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside
