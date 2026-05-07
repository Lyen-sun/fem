from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, ceil, hypot, sqrt
from time import perf_counter
from typing import Literal, Protocol

from fem_ai_solver.fem.model import Element, Mesh, Node
from fem_ai_solver.mesh.generation.constrained_delaunay import (
    BoundaryLoop,
    ConstrainedDelaunayInput,
    RegionMarker,
    generate_constrained_delaunay_t3,
)


Point2D = tuple[float, float]
MeshOperation = Literal["generate", "delete", "remesh"]
MeshBackendName = Literal["builtin", "gmsh"]
MeshAlgorithm = Literal["free", "structured"]


@dataclass(slots=True)
class MeshSeed:
    global_size: float = 1.0
    edge_seeds: dict[str, float] = field(default_factory=dict)
    region_sizes: dict[str, float] = field(default_factory=dict)
    # Reserved for Abaqus/Gmsh-like biased seeding. Builtin v1 keeps uniform seeds.
    bias: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class MeshControl:
    region_id: str | None = None
    algorithm: MeshAlgorithm = "free"
    element_type: Literal["T3"] = "T3"
    allow_boundary_preserve: bool = True


@dataclass(slots=True)
class MeshGeometryPoint:
    id: str
    x: float
    y: float
    owner_edge_id: str | None = None
    role: Literal["vertex", "edge_point", "interior_point"] = "vertex"


@dataclass(slots=True)
class MeshGeometryEdge:
    id: str
    start_point_id: str
    end_point_id: str


@dataclass(slots=True)
class MeshLoop:
    id: str
    points: list[Point2D]
    edge_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MeshRegion:
    id: str
    name: str
    points: list[Point2D]
    material_id: int = 1
    mesh_size: float | None = None
    hole_points: list[list[Point2D]] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    point_ids: list[str] = field(default_factory=list)
    component_id: str | None = None


@dataclass(slots=True)
class MeshQualityIssue:
    element_id: int
    kind: str
    value: float
    message: str


@dataclass(slots=True)
class MeshQualityReport:
    total_elements: int
    bad_element_ids: list[int]
    min_angle: float
    max_angle: float
    min_area: float
    max_area: float
    min_shape_quality: float
    max_aspect_ratio: float
    issues: list[MeshQualityIssue] = field(default_factory=list)

    @property
    def bad_element_count(self) -> int:
        return len(self.bad_element_ids)

    @property
    def summary(self) -> str:
        if self.total_elements <= 0:
            return "No T3 elements."
        return (
            f"T3 quality: elements={self.total_elements}, bad={self.bad_element_count}, "
            f"min_angle={self.min_angle:.2f}, max_angle={self.max_angle:.2f}, "
            f"min_quality={self.min_shape_quality:.3f}, max_aspect={self.max_aspect_ratio:.2f}"
        )


@dataclass(slots=True)
class MeshGenerationRequest:
    regions: list[MeshRegion]
    seed: MeshSeed = field(default_factory=MeshSeed)
    controls: list[MeshControl] = field(default_factory=list)
    geometry_points: dict[str, MeshGeometryPoint] = field(default_factory=dict)
    geometry_edges: dict[str, MeshGeometryEdge] = field(default_factory=dict)
    backend: MeshBackendName = "builtin"
    operation: MeshOperation = "generate"
    existing_mesh: Mesh | None = None
    target_region_ids: set[str] | None = None
    target_element_ids: set[int] | None = None
    preserve_node_ids: set[int] = field(default_factory=set)


@dataclass(slots=True)
class MeshGenerationResult:
    mesh: Mesh
    quality_report: MeshQualityReport
    region_to_element_ids: dict[str, list[int]] = field(default_factory=dict)
    edge_to_node_ids: dict[str, list[int]] = field(default_factory=dict)
    point_to_node_ids: dict[str, list[int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class MesherBackend(Protocol):
    name: MeshBackendName

    def generate(self, request: MeshGenerationRequest) -> MeshGenerationResult:
        ...


def polygon_area(points: list[Point2D]) -> float:
    if len(points) < 3:
        return 0.0
    twice = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        twice += x0 * y1 - x1 * y0
    return 0.5 * twice


def polygon_centroid(points: list[Point2D]) -> Point2D:
    area = polygon_area(points)
    if abs(area) <= 1e-12:
        if not points:
            return (0.0, 0.0)
        return (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))
    cx = 0.0
    cy = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    scale = 1.0 / (6.0 * area)
    return (cx * scale, cy * scale)


def point_on_segment(point: Point2D, a: Point2D, b: Point2D, *, tol: float = 1e-9) -> bool:
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
    return -tol <= dot <= seg_len2 + tol


def point_in_polygon(point: Point2D, polygon: list[Point2D]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for idx, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(idx + 1) % len(polygon)]
        if point_on_segment(point, (x0, y0), (x1, y1), tol=1e-9):
            return True
        if (y0 > y) == (y1 > y):
            continue
        xin = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        if xin >= x:
            inside = not inside
    return inside


def evaluate_t3_mesh_quality(
    mesh: Mesh,
    *,
    min_angle_limit: float = 15.0,
    max_angle_limit: float = 150.0,
    max_aspect_limit: float = 10.0,
    min_shape_quality_limit: float = 0.08,
    area_tolerance: float = 1e-14,
) -> MeshQualityReport:
    node_by_id = {int(node.id): node for node in mesh.nodes}
    issues: list[MeshQualityIssue] = []
    bad_ids: set[int] = set()
    min_angle = float("inf")
    max_angle = 0.0
    min_area = float("inf")
    max_area = 0.0
    min_quality = float("inf")
    max_aspect = 0.0
    total = 0

    for element in mesh.elements:
        if element.type != "T3" or len(element.connectivity) != 3:
            continue
        total += 1
        try:
            n1 = node_by_id[int(element.connectivity[0])]
            n2 = node_by_id[int(element.connectivity[1])]
            n3 = node_by_id[int(element.connectivity[2])]
        except KeyError:
            issues.append(MeshQualityIssue(int(element.id), "reference", 0.0, "Element references an unknown node."))
            bad_ids.add(int(element.id))
            continue

        p1 = (float(n1.x), float(n1.y))
        p2 = (float(n2.x), float(n2.y))
        p3 = (float(n3.x), float(n3.y))
        signed_twice_area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        area = 0.5 * signed_twice_area
        abs_area = abs(area)
        min_area = min(min_area, abs_area)
        max_area = max(max_area, abs_area)
        if abs_area <= area_tolerance:
            issues.append(MeshQualityIssue(int(element.id), "degenerate", area, "Degenerate near-zero-area T3 element."))
            bad_ids.add(int(element.id))
            continue
        if area < 0.0:
            issues.append(MeshQualityIssue(int(element.id), "orientation", area, "Clockwise T3 element orientation."))
            bad_ids.add(int(element.id))

        lengths = (
            hypot(p2[0] - p3[0], p2[1] - p3[1]),
            hypot(p1[0] - p3[0], p1[1] - p3[1]),
            hypot(p1[0] - p2[0], p1[1] - p2[1]),
        )
        shortest = max(min(lengths), 1e-30)
        longest = max(lengths)
        aspect = longest / shortest
        max_aspect = max(max_aspect, aspect)
        quality = max(0.0, 4.0 * sqrt(3.0) * abs_area / max(sum(length * length for length in lengths), 1e-30))
        min_quality = min(min_quality, quality)

        angles: list[float] = []
        for i in range(3):
            a = lengths[i]
            b = lengths[(i + 1) % 3]
            c = lengths[(i + 2) % 3]
            denom = max(2.0 * b * c, 1e-30)
            cos_theta = max(-1.0, min(1.0, (b * b + c * c - a * a) / denom))
            angles.append(acos(cos_theta) * 180.0 / 3.141592653589793)
        elem_min_angle = min(angles)
        elem_max_angle = max(angles)
        min_angle = min(min_angle, elem_min_angle)
        max_angle = max(max_angle, elem_max_angle)

        if elem_min_angle < min_angle_limit:
            issues.append(MeshQualityIssue(int(element.id), "small_angle", elem_min_angle, "Minimum angle below quality limit."))
            bad_ids.add(int(element.id))
        if elem_max_angle > max_angle_limit:
            issues.append(MeshQualityIssue(int(element.id), "large_angle", elem_max_angle, "Maximum angle above quality limit."))
            bad_ids.add(int(element.id))
        if aspect > max_aspect_limit:
            issues.append(MeshQualityIssue(int(element.id), "aspect_ratio", aspect, "Aspect ratio above quality limit."))
            bad_ids.add(int(element.id))
        if quality < min_shape_quality_limit:
            issues.append(MeshQualityIssue(int(element.id), "shape_quality", quality, "Shape quality below limit."))
            bad_ids.add(int(element.id))

    if total == 0:
        min_angle = 0.0
        min_area = 0.0
        min_quality = 0.0
    if min_area == float("inf"):
        min_area = 0.0
    if min_quality == float("inf"):
        min_quality = 0.0
    return MeshQualityReport(
        total_elements=total,
        bad_element_ids=sorted(bad_ids),
        min_angle=float(min_angle),
        max_angle=float(max_angle),
        min_area=float(min_area),
        max_area=float(max_area),
        min_shape_quality=float(min_quality),
        max_aspect_ratio=float(max_aspect),
        issues=issues,
    )


def delete_elements_from_mesh(mesh: Mesh, element_ids: set[int], *, preserve_node_ids: set[int] | None = None) -> Mesh:
    remove_ids = {int(item) for item in element_ids}
    preserve_ids = {int(item) for item in (preserve_node_ids or set())}
    kept_elements = [element for element in mesh.elements if int(element.id) not in remove_ids]
    referenced_nodes = {
        int(node_id)
        for element in kept_elements
        for node_id in element.connectivity
    }
    referenced_nodes.update(preserve_ids)
    kept_nodes = [node for node in mesh.nodes if int(node.id) in referenced_nodes]
    return Mesh(nodes=kept_nodes, elements=kept_elements)


def _canonical_coord(point: Point2D) -> tuple[float, float]:
    return (round(float(point[0]), 12), round(float(point[1]), 12))


def _normalize_polygon(points: list[Point2D]) -> list[Point2D]:
    normalized = [(float(x), float(y)) for x, y in points]
    if len(normalized) > 1 and hypot(normalized[0][0] - normalized[-1][0], normalized[0][1] - normalized[-1][1]) <= 1e-12:
        normalized = normalized[:-1]
    return normalized


def _edge_size(region: MeshRegion, seed: MeshSeed, edge_id: str | None) -> float:
    if edge_id and edge_id in seed.edge_seeds:
        return max(float(seed.edge_seeds[edge_id]), 1e-9)
    if region.id in seed.region_sizes:
        return max(float(seed.region_sizes[region.id]), 1e-9)
    if region.mesh_size is not None and float(region.mesh_size) > 0.0:
        return max(float(region.mesh_size), 1e-9)
    return max(float(seed.global_size), 1e-9)


def _densify_loop(points: list[Point2D], edge_ids: list[str], region: MeshRegion, seed: MeshSeed) -> tuple[list[Point2D], list[str | None]]:
    polygon = _normalize_polygon(points)
    if len(polygon) < 3:
        return [], []
    result: list[Point2D] = []
    source_edges: list[str | None] = []
    for idx, p0 in enumerate(polygon):
        p1 = polygon[(idx + 1) % len(polygon)]
        edge_id = edge_ids[idx] if idx < len(edge_ids) else None
        size = _edge_size(region, seed, edge_id)
        length = hypot(p1[0] - p0[0], p1[1] - p0[1])
        pieces = max(1, int(ceil(length / size)))
        for k in range(pieces):
            t = float(k) / float(pieces)
            point = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            if result and _canonical_coord(result[-1]) == _canonical_coord(point):
                continue
            result.append(point)
            source_edges.append(edge_id)
    return result, source_edges


def _region_step(region: MeshRegion, seed: MeshSeed) -> float:
    if region.id in seed.region_sizes:
        return max(float(seed.region_sizes[region.id]), 1e-9)
    if region.mesh_size is not None and float(region.mesh_size) > 0.0:
        return max(float(region.mesh_size), 1e-9)
    return max(float(seed.global_size), 1e-9)


def _control_by_region(controls: list[MeshControl]) -> dict[str | None, MeshControl]:
    return {control.region_id: control for control in controls}


def _extra_geometry_points_for_region(
    region: MeshRegion,
    geometry_points: dict[str, MeshGeometryPoint],
) -> dict[str, Point2D]:
    outer = _normalize_polygon(region.points)
    holes = [_normalize_polygon(item) for item in region.hole_points if len(_normalize_polygon(item)) >= 3]
    if len(outer) < 3:
        return {}
    result: dict[str, Point2D] = {}
    for point_id, point in geometry_points.items():
        xy = (float(point.x), float(point.y))
        if any(_canonical_coord(xy) == _canonical_coord(vertex) for vertex in outer):
            continue
        if any(any(_canonical_coord(xy) == _canonical_coord(vertex) for vertex in hole) for hole in holes):
            continue
        if not point_in_polygon(xy, outer):
            continue
        if any(point_in_polygon(xy, hole) for hole in holes):
            continue
        result[str(point_id)] = xy
    return result


def _interior_points(outer: list[Point2D], holes: list[list[Point2D]], step: float) -> list[Point2D]:
    if len(outer) < 3 or step <= 0.0:
        return []
    xs = [x for x, _ in outer]
    ys = [y for _, y in outer]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    nx = max(0, int(ceil((xmax - xmin) / step)))
    ny = max(0, int(ceil((ymax - ymin) / step)))
    if nx * ny > 30_000:
        scale = sqrt((nx * ny) / 30_000.0)
        step *= max(scale, 1.0)
        nx = max(0, int(ceil((xmax - xmin) / step)))
        ny = max(0, int(ceil((ymax - ymin) / step)))
    points: list[Point2D] = []
    for ix in range(nx):
        x = xmin + (ix + 0.5) * step
        if x <= xmin or x >= xmax:
            continue
        for iy in range(ny):
            y = ymin + (iy + 0.5) * step
            if y <= ymin or y >= ymax:
                continue
            point = (float(x), float(y))
            if not point_in_polygon(point, outer):
                continue
            if any(point_in_polygon(point, hole) for hole in holes):
                continue
            if any(point_on_segment(point, outer[idx], outer[(idx + 1) % len(outer)], tol=step * 1e-5) for idx in range(len(outer))):
                continue
            points.append(point)
    return points


class BuiltinT3Mesher:
    name: MeshBackendName = "builtin"

    def generate(self, request: MeshGenerationRequest) -> MeshGenerationResult:
        timings: dict[str, float] = {}
        warnings: list[str] = []
        t0 = perf_counter()
        if request.operation == "delete":
            if request.existing_mesh is None:
                raise ValueError("Delete mesh operation requires an existing mesh.")
            target_ids = {int(item) for item in (request.target_element_ids or set())}
            if not target_ids:
                raise ValueError("Delete mesh operation requires selected element ids.")
            mesh = delete_elements_from_mesh(
                request.existing_mesh,
                target_ids,
                preserve_node_ids=request.preserve_node_ids,
            )
            quality = evaluate_t3_mesh_quality(mesh)
            timings["total"] = (perf_counter() - t0) * 1000.0
            return MeshGenerationResult(mesh=mesh, quality_report=quality, warnings=warnings, timings_ms=timings)

        target_region_ids = set(request.target_region_ids or [])
        target_regions = [
            region
            for region in request.regions
            if not target_region_ids or region.id in target_region_ids
        ]
        if not target_regions:
            raise ValueError("No valid geometry regions are available for T3 meshing.")

        if request.operation == "remesh" and request.existing_mesh is not None and request.target_element_ids:
            base_mesh = delete_elements_from_mesh(
                request.existing_mesh,
                {int(item) for item in request.target_element_ids},
                preserve_node_ids=request.preserve_node_ids,
            )
        elif request.operation == "remesh" and request.existing_mesh is not None and target_region_ids:
            base_mesh = self._delete_region_elements(request.existing_mesh, request.regions, target_region_ids, request.preserve_node_ids)
        else:
            base_mesh = Mesh()

        control_map = _control_by_region(request.controls)
        for region in target_regions:
            control = control_map.get(region.id) or control_map.get(None)
            if control is not None and control.algorithm == "structured":
                warnings.append(
                    f"{region.name}: builtin backend currently uses geometry-preserving free T3 meshing for this region."
                )
        mesh, region_to_element_ids, edge_to_node_ids, point_to_node_ids = self._generate_regions(
            target_regions,
            request.seed,
            base_mesh,
            request.geometry_points,
        )
        timings["generate"] = (perf_counter() - t0) * 1000.0
        quality = evaluate_t3_mesh_quality(mesh)
        timings["quality"] = (perf_counter() - t0) * 1000.0 - timings["generate"]
        timings["total"] = (perf_counter() - t0) * 1000.0
        return MeshGenerationResult(
            mesh=mesh,
            quality_report=quality,
            region_to_element_ids=region_to_element_ids,
            edge_to_node_ids=edge_to_node_ids,
            point_to_node_ids=point_to_node_ids,
            warnings=warnings,
            timings_ms=timings,
        )

    def _delete_region_elements(
        self,
        mesh: Mesh,
        regions: list[MeshRegion],
        target_region_ids: set[str],
        preserve_node_ids: set[int],
    ) -> Mesh:
        node_by_id = {int(node.id): node for node in mesh.nodes}
        polygons = {
            region.id: _normalize_polygon(region.points)
            for region in regions
            if region.id in target_region_ids and len(_normalize_polygon(region.points)) >= 3
        }
        remove_ids: set[int] = set()
        for element in mesh.elements:
            if len(element.connectivity) != 3:
                continue
            nodes = [node_by_id.get(int(node_id)) for node_id in element.connectivity]
            if any(node is None for node in nodes):
                continue
            centroid = (
                (float(nodes[0].x) + float(nodes[1].x) + float(nodes[2].x)) / 3.0,
                (float(nodes[0].y) + float(nodes[1].y) + float(nodes[2].y)) / 3.0,
            )
            if any(point_in_polygon(centroid, polygon) for polygon in polygons.values()):
                remove_ids.add(int(element.id))
        return delete_elements_from_mesh(mesh, remove_ids, preserve_node_ids=preserve_node_ids)

    def _generate_regions(
        self,
        regions: list[MeshRegion],
        seed: MeshSeed,
        base_mesh: Mesh,
        geometry_points: dict[str, MeshGeometryPoint] | None = None,
    ) -> tuple[Mesh, dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
        geometry_points = geometry_points or {}
        nodes: list[Node] = [Node(id=int(node.id), x=float(node.x), y=float(node.y)) for node in base_mesh.nodes]
        elements: list[Element] = [
            Element(id=int(element.id), type=element.type, connectivity=list(element.connectivity), material_id=int(element.material_id))
            for element in base_mesh.elements
        ]
        node_id_by_coord = {_canonical_coord((node.x, node.y)): int(node.id) for node in nodes}
        next_node_id = max([node.id for node in nodes], default=0) + 1
        next_element_id = max([element.id for element in elements], default=0) + 1
        region_to_element_ids: dict[str, list[int]] = {}
        edge_to_node_ids: dict[str, list[int]] = {}
        point_to_node_ids: dict[str, list[int]] = {}

        def ensure_node(x: float, y: float) -> int:
            nonlocal next_node_id
            key = _canonical_coord((x, y))
            existing = node_id_by_coord.get(key)
            if existing is not None:
                return existing
            node_id = next_node_id
            next_node_id += 1
            node_id_by_coord[key] = node_id
            nodes.append(Node(id=node_id, x=key[0], y=key[1]))
            return node_id

        for region in regions:
            outer, source_edges = _densify_loop(region.points, region.edge_ids, region, seed)
            if len(outer) < 3:
                continue
            holes = [_normalize_polygon(item) for item in region.hole_points if len(_normalize_polygon(item)) >= 3]
            if polygon_area(outer) < 0.0:
                outer.reverse()
                source_edges.reverse()
            all_points: list[Point2D] = list(outer)
            outer_ids = list(range(len(all_points)))
            segments: list[tuple[int, int]] = []
            for idx in range(len(outer_ids)):
                segments.append((outer_ids[idx], outer_ids[(idx + 1) % len(outer_ids)]))
            hole_loops: list[BoundaryLoop] = []
            for hole in holes:
                if polygon_area(hole) > 0.0:
                    hole = list(reversed(hole))
                start = len(all_points)
                all_points.extend(hole)
                ids = list(range(start, len(all_points)))
                for idx in range(len(ids)):
                    segments.append((ids[idx], ids[(idx + 1) % len(ids)]))
                hole_loops.append(BoundaryLoop(point_ids=ids))

            embedded_start = len(all_points)
            extra_point_local_ids: dict[str, int] = {}
            for point_id, xy in _extra_geometry_points_for_region(region, geometry_points).items():
                if any(_canonical_coord(xy) == _canonical_coord(existing) for existing in all_points):
                    continue
                extra_point_local_ids[point_id] = len(all_points)
                all_points.append(xy)
            interior = _interior_points(outer, holes, _region_step(region, seed))
            all_points.extend(interior)
            embedded_ids = list(range(embedded_start, len(all_points)))
            centroid = polygon_centroid(outer)
            cdt_result = generate_constrained_delaunay_t3(
                ConstrainedDelaunayInput(
                    points=all_points,
                    segments=segments,
                    outer_loop=BoundaryLoop(point_ids=outer_ids),
                    holes=hole_loops,
                    regions=[RegionMarker(x=centroid[0], y=centroid[1], material_id=max(int(region.material_id), 1))],
                    embedded_point_ids=embedded_ids,
                )
            )

            local_to_global: dict[int, int] = {}
            for node in cdt_result.mesh.nodes:
                local_to_global[int(node.id)] = ensure_node(float(node.x), float(node.y))
            new_element_ids: list[int] = []
            for element in cdt_result.mesh.elements:
                conn = [local_to_global[int(node_id)] for node_id in element.connectivity]
                elements.append(
                    Element(
                        id=next_element_id,
                        type="T3",
                        connectivity=conn,
                        material_id=max(int(region.material_id), 1),
                    )
                )
                new_element_ids.append(next_element_id)
                next_element_id += 1
            region_to_element_ids[region.id] = new_element_ids

            for idx, edge_id in enumerate(source_edges):
                if edge_id is None:
                    continue
                node_id = local_to_global.get(idx + 1)
                if node_id is not None:
                    edge_to_node_ids.setdefault(edge_id, []).append(node_id)
            for idx in range(len(outer)):
                node_id = local_to_global.get(idx + 1)
                if node_id is not None:
                    point_to_node_ids[f"{region.id}:point:{idx}"] = [node_id]
            for idx, point_id in enumerate(region.point_ids):
                if idx >= len(_normalize_polygon(region.points)):
                    continue
                key = _canonical_coord(_normalize_polygon(region.points)[idx])
                node_id = node_id_by_coord.get(key)
                if node_id is not None:
                    point_to_node_ids[str(point_id)] = [int(node_id)]
            for point_id, local_idx in extra_point_local_ids.items():
                node_id = local_to_global.get(local_idx + 1)
                if node_id is not None:
                    point_to_node_ids[str(point_id)] = [int(node_id)]
            for edge_id, node_ids in list(edge_to_node_ids.items()):
                edge_to_node_ids[edge_id] = sorted(set(node_ids))

        return (
            Mesh(nodes=sorted(nodes, key=lambda item: item.id), elements=sorted(elements, key=lambda item: item.id)),
            region_to_element_ids,
            edge_to_node_ids,
            point_to_node_ids,
        )


class GmshMesher:
    name: MeshBackendName = "gmsh"

    def generate(self, request: MeshGenerationRequest) -> MeshGenerationResult:
        try:
            import gmsh  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "Gmsh Python package is not installed. Install 'gmsh' to use the Gmsh mesher, "
                "or switch the mesh backend to Builtin T3."
            ) from exc

        timings: dict[str, float] = {}
        warnings: list[str] = []
        t0 = perf_counter()
        if request.operation == "delete":
            if request.existing_mesh is None:
                raise ValueError("Delete mesh operation requires an existing mesh.")
            mesh = delete_elements_from_mesh(
                request.existing_mesh,
                {int(item) for item in (request.target_element_ids or set())},
                preserve_node_ids=request.preserve_node_ids,
            )
            quality = evaluate_t3_mesh_quality(mesh)
            timings["total"] = (perf_counter() - t0) * 1000.0
            return MeshGenerationResult(mesh=mesh, quality_report=quality, warnings=warnings, timings_ms=timings)

        target_region_ids = set(request.target_region_ids or [])
        if request.operation == "remesh" and request.existing_mesh is not None:
            regions = list(request.regions)
            if target_region_ids or request.target_element_ids:
                warnings.append("Gmsh backend currently regenerates the full geometry for remesh to keep geometry mappings stable.")
        else:
            regions = [region for region in request.regions if not target_region_ids or region.id in target_region_ids]
        if not regions:
            raise ValueError("No valid geometry regions are available for Gmsh T3 meshing.")

        control_map = _control_by_region(request.controls)
        was_initialized = bool(getattr(gmsh, "isInitialized", lambda: 0)())
        if not was_initialized:
            gmsh.initialize()
        try:
            gmsh.model.add(f"rockfem_mesh_{int(t0 * 1000)}")
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.option.setNumber("Mesh.ElementOrder", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 6)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            except Exception:
                pass

            point_tag_by_id: dict[str, int] = {}
            point_tag_by_coord: dict[tuple[float, float], int] = {}
            point_xy_by_id: dict[str, Point2D] = {}
            line_tag_by_pair: dict[tuple[int, int], int] = {}
            signed_line_by_edge_id: dict[str, int] = {}
            edge_endpoints: dict[str, tuple[Point2D, Point2D]] = {}
            surface_by_region_id: dict[str, int] = {}
            region_by_surface: dict[int, MeshRegion] = {}
            embedded_surface_points: list[tuple[int, int]] = []
            embedded_curve_points: list[tuple[int, int]] = []

            def add_point(point_id: str, xy: Point2D, size: float) -> int:
                key = _canonical_coord(xy)
                tag = point_tag_by_coord.get(key)
                if tag is None:
                    tag = int(gmsh.model.geo.addPoint(float(key[0]), float(key[1]), 0.0, max(float(size), 1e-9)))
                    point_tag_by_coord[key] = tag
                point_tag_by_id[str(point_id)] = tag
                point_xy_by_id[str(point_id)] = (float(key[0]), float(key[1]))
                return tag

            for point in request.geometry_points.values():
                add_point(str(point.id), (float(point.x), float(point.y)), request.seed.global_size)

            def add_line(start_tag: int, end_tag: int) -> int:
                existing = line_tag_by_pair.get((start_tag, end_tag))
                if existing is not None:
                    return existing
                reversed_existing = line_tag_by_pair.get((end_tag, start_tag))
                if reversed_existing is not None:
                    return -reversed_existing
                tag = int(gmsh.model.geo.addLine(start_tag, end_tag))
                line_tag_by_pair[(start_tag, end_tag)] = tag
                return tag

            for region in regions:
                points = _normalize_polygon(region.points)
                if len(points) < 3:
                    continue
                point_ids = list(region.point_ids)
                if len(point_ids) != len(points):
                    point_ids = [f"{region.id}:point:{idx}" for idx in range(len(points))]
                region_size = _region_step(region, request.seed)
                curve_tags: list[int] = []
                point_tags: list[int] = []
                for point_id, xy in zip(point_ids, points):
                    point_tags.append(add_point(str(point_id), xy, region_size))
                for idx, start_tag in enumerate(point_tags):
                    end_tag = point_tags[(idx + 1) % len(point_tags)]
                    line_tag = add_line(start_tag, end_tag)
                    curve_tags.append(line_tag)
                    edge_id = region.edge_ids[idx] if idx < len(region.edge_ids) else f"{region.id}:edge:{idx}"
                    signed_line_by_edge_id[str(edge_id)] = line_tag
                    p0 = points[idx]
                    p1 = points[(idx + 1) % len(points)]
                    edge_endpoints[str(edge_id)] = (p0, p1)
                    edge_size = _edge_size(region, request.seed, str(edge_id))
                    pieces = max(1, int(ceil(hypot(p1[0] - p0[0], p1[1] - p0[1]) / edge_size)))
                    if str(edge_id) in request.seed.edge_seeds or (control_map.get(region.id) or control_map.get(None)):
                        try:
                            gmsh.model.geo.mesh.setTransfiniteCurve(abs(line_tag), pieces + 1)
                        except Exception:
                            pass
                loop_tag = int(gmsh.model.geo.addCurveLoop(curve_tags))
                hole_loop_tags: list[int] = []
                for hole_idx, hole_raw in enumerate(region.hole_points):
                    hole = _normalize_polygon(hole_raw)
                    if len(hole) < 3:
                        continue
                    if polygon_area(hole) > 0.0:
                        hole = list(reversed(hole))
                    hole_point_tags = [
                        add_point(f"{region.id}:hole:{hole_idx}:point:{idx}", xy, region_size)
                        for idx, xy in enumerate(hole)
                    ]
                    hole_curves = [
                        add_line(hole_point_tags[idx], hole_point_tags[(idx + 1) % len(hole_point_tags)])
                        for idx in range(len(hole_point_tags))
                    ]
                    hole_loop_tags.append(int(gmsh.model.geo.addCurveLoop(hole_curves)))
                surface_tag = int(gmsh.model.geo.addPlaneSurface([loop_tag, *hole_loop_tags]))
                surface_by_region_id[region.id] = surface_tag
                region_by_surface[surface_tag] = region
                control = control_map.get(region.id) or control_map.get(None)
                if control is not None and control.algorithm == "structured":
                    if len(curve_tags) == 4 and not hole_loop_tags:
                        try:
                            gmsh.model.geo.mesh.setTransfiniteSurface(surface_tag)
                        except Exception as exc:
                            warnings.append(f"{region.name}: Gmsh transfinite surface setup failed ({exc}).")
                    else:
                        warnings.append(f"{region.name}: structured control requires a 4-edge region without holes; using free T3.")

            for point in request.geometry_points.values():
                tag = add_point(str(point.id), (float(point.x), float(point.y)), request.seed.global_size)
                if point.owner_edge_id and point.owner_edge_id in signed_line_by_edge_id:
                    embedded_curve_points.append((tag, abs(signed_line_by_edge_id[point.owner_edge_id])))
                    continue
                xy = (float(point.x), float(point.y))
                for region in regions:
                    outer = _normalize_polygon(region.points)
                    holes = [_normalize_polygon(item) for item in region.hole_points if len(_normalize_polygon(item)) >= 3]
                    if point_in_polygon(xy, outer) and not any(point_in_polygon(xy, hole) for hole in holes):
                        surface_tag = surface_by_region_id.get(region.id)
                        if surface_tag is not None:
                            embedded_surface_points.append((tag, surface_tag))
                        break

            gmsh.model.geo.synchronize()
            for point_tag, line_tag in embedded_curve_points:
                try:
                    gmsh.model.mesh.embed(0, [point_tag], 1, line_tag)
                except Exception as exc:
                    warnings.append(f"Gmsh could not embed a geometry point on curve {line_tag}: {exc}")
            for point_tag, surface_tag in embedded_surface_points:
                try:
                    gmsh.model.mesh.embed(0, [point_tag], 2, surface_tag)
                except Exception as exc:
                    warnings.append(f"Gmsh could not embed a geometry point in region {surface_tag}: {exc}")
            for point_id, tag in point_tag_by_id.items():
                try:
                    group = int(gmsh.model.addPhysicalGroup(0, [tag]))
                    gmsh.model.setPhysicalName(0, group, f"point:{point_id}")
                except Exception:
                    pass
            for edge_id, signed_tag in signed_line_by_edge_id.items():
                try:
                    group = int(gmsh.model.addPhysicalGroup(1, [abs(signed_tag)]))
                    gmsh.model.setPhysicalName(1, group, f"edge:{edge_id}")
                except Exception:
                    pass
            for region_id, surface_tag in surface_by_region_id.items():
                try:
                    group = int(gmsh.model.addPhysicalGroup(2, [surface_tag]))
                    gmsh.model.setPhysicalName(2, group, f"region:{region_id}")
                except Exception:
                    pass

            t_generate_start = perf_counter()
            gmsh.model.mesh.generate(2)
            timings["gmsh_generate"] = (perf_counter() - t_generate_start) * 1000.0

            node_tags_raw, coords_raw, _ = gmsh.model.mesh.getNodes()
            node_tags = [int(tag) for tag in list(node_tags_raw)]
            coords = [float(value) for value in list(coords_raw)]
            coord_by_tag = {
                tag: (coords[3 * idx], coords[3 * idx + 1])
                for idx, tag in enumerate(node_tags)
            }
            node_id_by_tag = {tag: idx + 1 for idx, tag in enumerate(sorted(coord_by_tag))}
            nodes = [
                Node(id=node_id_by_tag[tag], x=float(coord_by_tag[tag][0]), y=float(coord_by_tag[tag][1]))
                for tag in sorted(coord_by_tag)
            ]
            extent = 1.0
            if nodes:
                xs = [node.x for node in nodes]
                ys = [node.y for node in nodes]
                extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            tol = max(extent * 1e-7, 1e-9)

            elements: list[Element] = []
            region_to_element_ids: dict[str, list[int]] = {}
            for region_id, surface_tag in surface_by_region_id.items():
                region = region_by_surface[surface_tag]
                region_elements: list[int] = []
                element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(2, surface_tag)
                for element_type, _, flat_nodes in zip(element_types, element_tags, element_node_tags):
                    if int(element_type) != 2:
                        continue
                    flat = [int(tag) for tag in list(flat_nodes)]
                    for idx in range(0, len(flat), 3):
                        tri_tags = flat[idx: idx + 3]
                        if len(tri_tags) != 3 or any(tag not in node_id_by_tag for tag in tri_tags):
                            continue
                        conn = [node_id_by_tag[tag] for tag in tri_tags]
                        p1 = coord_by_tag[tri_tags[0]]
                        p2 = coord_by_tag[tri_tags[1]]
                        p3 = coord_by_tag[tri_tags[2]]
                        signed_two_area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                        if signed_two_area < 0.0:
                            conn[1], conn[2] = conn[2], conn[1]
                        element_id = len(elements) + 1
                        elements.append(
                            Element(
                                id=element_id,
                                type="T3",
                                connectivity=conn,
                                material_id=max(int(region.material_id), 1),
                            )
                        )
                        region_elements.append(element_id)
                region_to_element_ids[region_id] = region_elements

            def closest_node_ids(xy: Point2D) -> list[int]:
                matches: list[tuple[float, int]] = []
                for tag, coord in coord_by_tag.items():
                    distance = hypot(coord[0] - xy[0], coord[1] - xy[1])
                    if distance <= tol:
                        matches.append((distance, node_id_by_tag[tag]))
                if matches:
                    matches.sort(key=lambda item: (item[0], item[1]))
                    return [int(matches[0][1])]
                if not coord_by_tag:
                    return []
                tag = min(coord_by_tag, key=lambda item: hypot(coord_by_tag[item][0] - xy[0], coord_by_tag[item][1] - xy[1]))
                if hypot(coord_by_tag[tag][0] - xy[0], coord_by_tag[tag][1] - xy[1]) <= max(tol * 50.0, 1e-6):
                    return [int(node_id_by_tag[tag])]
                return []

            point_to_node_ids = {
                point_id: closest_node_ids((float(point.x), float(point.y)))
                for point_id, point in request.geometry_points.items()
            }
            for region in regions:
                for idx, point_id in enumerate(region.point_ids):
                    if idx < len(region.points):
                        point_to_node_ids.setdefault(str(point_id), closest_node_ids(region.points[idx]))

            def ordered_edge_nodes(edge_id: str, p0: Point2D, p1: Point2D) -> list[int]:
                candidates: set[int] = set()
                line_tag = signed_line_by_edge_id.get(edge_id)
                if line_tag is not None:
                    try:
                        line_node_tags, _, _ = gmsh.model.mesh.getNodes(1, abs(line_tag), True)
                        candidates.update(node_id_by_tag[int(tag)] for tag in list(line_node_tags) if int(tag) in node_id_by_tag)
                    except Exception:
                        pass
                for tag, coord in coord_by_tag.items():
                    if point_on_segment(coord, p0, p1, tol=tol * 5.0):
                        candidates.add(node_id_by_tag[tag])
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                denom = max(dx * dx + dy * dy, 1e-30)
                node_by_id = {node.id: node for node in nodes}
                return sorted(
                    candidates,
                    key=lambda node_id: (
                        ((node_by_id[node_id].x - p0[0]) * dx + (node_by_id[node_id].y - p0[1]) * dy) / denom,
                        node_id,
                    ),
                )

            edge_to_node_ids: dict[str, list[int]] = {}
            for edge_id, edge in request.geometry_edges.items():
                start = request.geometry_points.get(edge.start_point_id)
                end = request.geometry_points.get(edge.end_point_id)
                if start is None or end is None:
                    continue
                edge_to_node_ids[edge_id] = ordered_edge_nodes(edge_id, (float(start.x), float(start.y)), (float(end.x), float(end.y)))
            for edge_id, (p0, p1) in edge_endpoints.items():
                edge_to_node_ids.setdefault(edge_id, ordered_edge_nodes(edge_id, p0, p1))

            mesh = Mesh(nodes=nodes, elements=elements)
            timings["convert"] = (perf_counter() - t0) * 1000.0 - timings.get("gmsh_generate", 0.0)
            quality = evaluate_t3_mesh_quality(mesh)
            timings["total"] = (perf_counter() - t0) * 1000.0
            return MeshGenerationResult(
                mesh=mesh,
                quality_report=quality,
                region_to_element_ids=region_to_element_ids,
                edge_to_node_ids=edge_to_node_ids,
                point_to_node_ids=point_to_node_ids,
                warnings=warnings,
                timings_ms=timings,
            )
        finally:
            if not was_initialized:
                gmsh.finalize()


def mesher_backend(name: str) -> MesherBackend:
    if str(name).strip().lower() == "gmsh":
        return GmshMesher()
    return BuiltinT3Mesher()
