from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from fem_ai_solver.fem.model import Element, Mesh, Node
from fem_ai_solver.mesh.generation.github_ear_clipping import triangulate_polygon_ear_clipping


@dataclass(slots=True)
class BoundaryLoop:
    point_ids: list[int]


@dataclass(slots=True)
class RegionMarker:
    x: float
    y: float
    material_id: int


@dataclass(slots=True)
class ConstrainedDelaunayInput:
    points: list[tuple[float, float]]
    segments: list[tuple[int, int]]
    outer_loop: BoundaryLoop
    holes: list[BoundaryLoop]
    regions: list[RegionMarker]
    embedded_point_ids: list[int] | None = None


@dataclass(slots=True)
class ConstrainedDelaunayResult:
    mesh: Mesh
    input_point_to_node_id: dict[int, int]


@dataclass(slots=True)
class _BoundarySegment:
    p0: tuple[float, float]
    p1: tuple[float, float]
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


def _point_on_segment(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    tol: float = 1e-10,
) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= tol * tol:
        return hypot(px - ax, py - ay) <= tol
    cross = (px - ax) * dy - (py - ay) * dx
    if abs(cross) > tol * max(1.0, hypot(dx, dy)):
        return False
    dot = (px - ax) * dx + (py - ay) * dy
    if dot < -tol:
        return False
    if dot > seg_len2 + tol:
        return False
    return True


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        if _point_on_segment((x, y), (x0, y0), (x1, y1), tol=1e-9):
            return True
        cond = (y0 > y) != (y1 > y)
        if not cond:
            continue
        xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        if xin >= x:
            inside = not inside
    return inside


def _segments_intersect(
    p0: tuple[float, float],
    p1: tuple[float, float],
    q0: tuple[float, float],
    q1: tuple[float, float],
    *,
    tol: float = 1e-12,
) -> bool:
    def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p0, p1, q0)
    o2 = orient(p0, p1, q1)
    o3 = orient(q0, q1, p0)
    o4 = orient(q0, q1, p1)
    if (o1 > tol and o2 < -tol or o1 < -tol and o2 > tol) and (o3 > tol and o4 < -tol or o3 < -tol and o4 > tol):
        return True
    if abs(o1) <= tol and _point_on_segment(q0, p0, p1, tol=1e-9):
        return True
    if abs(o2) <= tol and _point_on_segment(q1, p0, p1, tol=1e-9):
        return True
    if abs(o3) <= tol and _point_on_segment(p0, q0, q1, tol=1e-9):
        return True
    if abs(o4) <= tol and _point_on_segment(p1, q0, q1, tol=1e-9):
        return True
    return False


def _loop_points(points: list[tuple[float, float]], loop: BoundaryLoop) -> list[tuple[float, float]]:
    if len(loop.point_ids) < 3:
        return []
    seq: list[tuple[float, float]] = []
    for idx in loop.point_ids:
        if idx < 0 or idx >= len(points):
            return []
        seq.append((float(points[idx][0]), float(points[idx][1])))
    return seq


def _assign_material_id(
    centroid: tuple[float, float],
    region_markers: list[RegionMarker],
    default_material_id: int,
) -> int:
    if not region_markers:
        return int(default_material_id)
    cx, cy = centroid
    best = region_markers[0]
    best_d2 = (cx - float(best.x)) ** 2 + (cy - float(best.y)) ** 2
    for marker in region_markers[1:]:
        d2 = (cx - float(marker.x)) ** 2 + (cy - float(marker.y)) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = marker
    return int(best.material_id)


def _triangle_crosses_boundary(
    tri: tuple[int, int, int],
    points: list[tuple[float, float]],
    boundaries: list[_BoundarySegment],
) -> bool:
    p = [points[tri[0]], points[tri[1]], points[tri[2]]]
    tri_edges = [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])]
    tri_vertex_set = {p[0], p[1], p[2]}
    tri_xmin = min(point[0] for point in p)
    tri_xmax = max(point[0] for point in p)
    tri_ymin = min(point[1] for point in p)
    tri_ymax = max(point[1] for point in p)
    for a, b in tri_edges:
        edge_xmin = min(a[0], b[0], tri_xmin)
        edge_xmax = max(a[0], b[0], tri_xmax)
        edge_ymin = min(a[1], b[1], tri_ymin)
        edge_ymax = max(a[1], b[1], tri_ymax)
        for boundary in boundaries:
            if boundary.xmax < edge_xmin or boundary.xmin > edge_xmax or boundary.ymax < edge_ymin or boundary.ymin > edge_ymax:
                continue
            q0 = boundary.p0
            q1 = boundary.p1
            # Shared endpoints are allowed.
            if a in (q0, q1) or b in (q0, q1):
                continue
            if _segments_intersect(a, b, q0, q1):
                return True
            # Boundary segment fully passing through triangle without endpoint touch.
            if q0 not in tri_vertex_set and q1 not in tri_vertex_set:
                if _point_on_segment(q0, a, b, tol=1e-9) or _point_on_segment(q1, a, b, tol=1e-9):
                    return True
    return False


def generate_constrained_delaunay_t3(payload: ConstrainedDelaunayInput) -> ConstrainedDelaunayResult:
    points = [(float(x), float(y)) for x, y in payload.points]
    if len(points) < 3:
        raise ValueError("Constrained Delaunay requires at least 3 input points.")
    outer_polygon = _loop_points(points, payload.outer_loop)
    if len(outer_polygon) < 3:
        raise ValueError("Outer boundary loop is invalid.")
    hole_polygons = [polygon for polygon in (_loop_points(points, loop) for loop in payload.holes) if len(polygon) >= 3]

    # Ensure consistent boundary orientation for robust inside tests.
    if _polygon_area(outer_polygon) < 0.0:
        outer_polygon = list(reversed(outer_polygon))
    normalized_holes: list[list[tuple[float, float]]] = []
    for polygon in hole_polygons:
        if _polygon_area(polygon) > 0.0:
            normalized_holes.append(list(reversed(polygon)))
        else:
            normalized_holes.append(polygon)

    boundaries: list[_BoundarySegment] = []
    loops = [outer_polygon, *normalized_holes]
    for polygon in loops:
        for i, p0 in enumerate(polygon):
            p1 = polygon[(i + 1) % len(polygon)]
            boundaries.append(
                _BoundarySegment(
                    p0=p0,
                    p1=p1,
                    xmin=min(p0[0], p1[0]),
                    xmax=max(p0[0], p1[0]),
                    ymin=min(p0[1], p1[1]),
                    ymax=max(p0[1], p1[1]),
                )
            )

    try:
        from scipy.spatial import Delaunay
    except Exception as exc:
        # Lightweight fallback for environments without scipy.
        if normalized_holes:
            raise RuntimeError("scipy is required for constrained meshing with holes.") from exc
        if payload.embedded_point_ids:
            raise RuntimeError("scipy is required to preserve embedded interior points.") from exc
        # Ear clipping fallback only supports pure outer-loop polygons.
        index_by_point = {points[idx]: idx for idx in payload.outer_loop.point_ids}
        if len(index_by_point) != len(payload.outer_loop.point_ids):
            raise RuntimeError("Ear clipping fallback does not support duplicate outer-loop vertices.") from exc
        local_poly = [points[idx] for idx in payload.outer_loop.point_ids]
        local_tris = triangulate_polygon_ear_clipping(local_poly)
        elements: list[Element] = []
        default_mid = int(payload.regions[0].material_id) if payload.regions else 1
        for elem_id, tri in enumerate(local_tris, start=1):
            conn = [payload.outer_loop.point_ids[tri[0]] + 1, payload.outer_loop.point_ids[tri[1]] + 1, payload.outer_loop.point_ids[tri[2]] + 1]
            elements.append(Element(id=elem_id, type="T3", connectivity=conn, material_id=default_mid))
        mesh = Mesh(nodes=[Node(id=i + 1, x=xy[0], y=xy[1]) for i, xy in enumerate(points)], elements=elements)
        return ConstrainedDelaunayResult(
            mesh=mesh,
            input_point_to_node_id={idx: idx + 1 for idx in range(len(points))},
        )

    tri = Delaunay(points)
    simplices = [tuple(int(v) for v in simplex) for simplex in tri.simplices]

    default_mid = int(payload.regions[0].material_id) if payload.regions else 1
    elements: list[Element] = []
    for simplex in simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        centroid = ((p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0)
        if not _point_in_polygon(centroid, outer_polygon):
            continue
        if any(_point_in_polygon(centroid, hole) for hole in normalized_holes):
            continue
        if _triangle_crosses_boundary(simplex, points, boundaries):
            continue
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if area <= 1e-14:
            simplex = (simplex[0], simplex[2], simplex[1])
        material_id = _assign_material_id(centroid, payload.regions, default_mid)
        elements.append(
            Element(
                id=len(elements) + 1,
                type="T3",
                connectivity=[simplex[0] + 1, simplex[1] + 1, simplex[2] + 1],
                material_id=material_id,
            )
        )

    if not elements:
        raise RuntimeError("Constrained Delaunay produced no valid T3 elements for current input.")

    mesh = Mesh(
        nodes=[Node(id=idx + 1, x=xy[0], y=xy[1]) for idx, xy in enumerate(points)],
        elements=elements,
    )
    return ConstrainedDelaunayResult(
        mesh=mesh,
        input_point_to_node_id={idx: idx + 1 for idx in range(len(points))},
    )
