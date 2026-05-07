from __future__ import annotations

"""Polygon triangulation based on GitHub-sourced ear clipping implementation.

Adapted from:
    https://github.com/joelibaceta/triangulator

Attribution:
    Original project metadata declares MIT license in setup.py.
    This module keeps the same algorithm idea, rewritten to return index-based
    triangles for direct FEM connectivity construction.
"""

from collections.abc import Sequence


Point2D = tuple[float, float]
TriangleIndex = tuple[int, int, int]


def _signed_area(points: Sequence[Point2D]) -> float:
    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _cross(o: Point2D, a: Point2D, b: Point2D) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(a: Point2D, b: Point2D, c: Point2D, p: Point2D) -> bool:
    c1 = _cross(a, b, p)
    c2 = _cross(b, c, p)
    c3 = _cross(c, a, p)
    has_neg = (c1 < 0.0) or (c2 < 0.0) or (c3 < 0.0)
    has_pos = (c1 > 0.0) or (c2 > 0.0) or (c3 > 0.0)
    return not (has_neg and has_pos)


def triangulate_polygon_ear_clipping(points: Sequence[Point2D]) -> list[TriangleIndex]:
    """Triangulate a simple polygon into CCW triangles.

    Args:
        points: Polygon vertices in order (clockwise or counterclockwise).

    Returns:
        Triangles as index tuples referencing the original ``points`` sequence.
    """
    if len(points) < 3:
        raise ValueError("Polygon needs at least 3 points.")

    pts = [(float(x), float(y)) for x, y in points]
    indices = list(range(len(pts)))
    if _signed_area(pts) < 0.0:
        indices.reverse()

    triangles: list[TriangleIndex] = []
    guard = 0
    while len(indices) > 3:
        guard += 1
        if guard > len(pts) * len(pts):
            raise ValueError("Ear clipping failed: polygon may be self-intersecting or degenerate.")

        ear_found = False
        for i in range(len(indices)):
            i_prev = indices[(i - 1) % len(indices)]
            i_curr = indices[i]
            i_next = indices[(i + 1) % len(indices)]

            a = pts[i_prev]
            b = pts[i_curr]
            c = pts[i_next]

            if _cross(a, b, c) <= 1e-14:
                continue

            is_ear = True
            for j in indices:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle(a, b, c, pts[j]):
                    is_ear = False
                    break
            if not is_ear:
                continue

            triangles.append((i_prev, i_curr, i_next))
            indices.pop(i)
            ear_found = True
            break

        if not ear_found:
            raise ValueError("Ear clipping failed: no valid ear found.")

    triangles.append((indices[0], indices[1], indices[2]))
    return triangles

