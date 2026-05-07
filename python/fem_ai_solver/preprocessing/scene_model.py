from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from fem_ai_solver.fem.model import Model

if TYPE_CHECKING:
    from fem_ai_solver.mesh.workflow import MeshQualityReport


SceneSource = Literal["sketch", "import", "ai", "manual"]
GeometryEntityType = Literal["point", "edge", "region", "component"]
GeometryPointRole = Literal["vertex", "edge_point", "interior_point"]
TargetBindingMode = Literal["mesh", "geometry"]
SceneLoadType = Literal["concentrated", "distributed", "pressure", "body", "gravity"]
SceneLoadProfile = Literal["uniform", "linear"]
SceneLoadDirectionMode = Literal["vector", "normal", "reverse_normal", "gravity"]
SceneBoundaryDirection = Literal["x", "y", "xy"]


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    twice = 0.0
    count = len(points)
    for idx in range(count):
        x0, y0 = points[idx]
        x1, y1 = points[(idx + 1) % count]
        twice += x0 * y1 - x1 * y0
    return 0.5 * twice


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 3:
        return (0.0, 0.0)
    area = polygon_area(points)
    if abs(area) <= 1e-12:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (sum(xs) / max(len(xs), 1), sum(ys) / max(len(ys), 1))
    c_x = 0.0
    c_y = 0.0
    count = len(points)
    for idx in range(count):
        x0, y0 = points[idx]
        x1, y1 = points[(idx + 1) % count]
        cross = x0 * y1 - x1 * y0
        c_x += (x0 + x1) * cross
        c_y += (y0 + y1) * cross
    scale = 1.0 / (6.0 * area)
    return (c_x * scale, c_y * scale)


@dataclass(slots=True)
class LoopDef:
    id: str
    name: str
    points: list[tuple[float, float]]
    hint: dict[str, object] | None = None
    is_closed: bool = True
    is_valid: bool = True
    source: SceneSource = "sketch"


@dataclass(slots=True)
class RegionDef:
    id: str
    name: str
    loop_ids: list[str]
    hole_loop_ids: list[str] = field(default_factory=list)
    component_id: str = "component:default"
    material_id: int | None = None
    mesh_size: float | None = None
    display_color: str = "#93c5fd"
    visible: bool = True
    area: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)
    source: SceneSource = "sketch"


@dataclass(slots=True)
class ComponentDef:
    id: str
    name: str
    region_ids: list[str] = field(default_factory=list)
    display_color: str = "#2563eb"
    visible: bool = True
    isolated: bool = False
    locked: bool = False


@dataclass(slots=True)
class GeometrySetDef:
    id: str
    name: str
    entity_type: GeometryEntityType
    entity_ids: list[str] = field(default_factory=list)
    binding_mode: TargetBindingMode = "mesh"
    auto_update: bool = True
    display_color: str = "#f59e0b"


@dataclass(slots=True)
class LoadDefinition:
    id: str
    name: str
    target_set_id: str
    target_entity_type: GeometryEntityType
    load_type: SceneLoadType
    vector_x: float
    vector_y: float
    magnitude: float
    profile: SceneLoadProfile = "uniform"
    step: str = "Initial"
    direction_mode: SceneLoadDirectionMode = "vector"
    active: bool = True


@dataclass(slots=True)
class BoundaryDefinition:
    id: str
    name: str
    target_set_id: str
    target_entity_type: GeometryEntityType
    direction: SceneBoundaryDirection
    value: float = 0.0
    step: str = "Initial"
    boundary_type: str = "displacement"
    active: bool = True


@dataclass(slots=True)
class GeometryPointDef:
    id: str
    name: str
    x: float
    y: float
    owner_loop_id: str | None = None
    owner_edge_id: str | None = None
    param: float | None = None
    source: SceneSource = "sketch"
    role: GeometryPointRole = "vertex"


@dataclass(slots=True)
class GeometryEdgeDef:
    id: str
    name: str
    loop_id: str
    start_point_id: str
    end_point_id: str
    source: SceneSource = "sketch"


@dataclass(slots=True)
class SceneMeshState:
    solver_model: Model | None = None
    point_to_node_ids: dict[str, list[int]] = field(default_factory=dict)
    edge_to_node_ids: dict[str, list[int]] = field(default_factory=dict)
    region_to_element_ids: dict[str, list[int]] = field(default_factory=dict)
    component_to_element_ids: dict[str, list[int]] = field(default_factory=dict)
    set_to_node_ids: dict[str, list[int]] = field(default_factory=dict)
    set_to_element_ids: dict[str, list[int]] = field(default_factory=dict)
    quality_report: MeshQualityReport | None = None
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SceneSelection:
    active_component_id: str | None = None
    active_region_ids: set[str] = field(default_factory=set)
    active_set_id: str | None = None


@dataclass(slots=True)
class SceneProject:
    loops: dict[str, LoopDef] = field(default_factory=dict)
    geometry_points: dict[str, GeometryPointDef] = field(default_factory=dict)
    geometry_edges: dict[str, GeometryEdgeDef] = field(default_factory=dict)
    regions: dict[str, RegionDef] = field(default_factory=dict)
    components: dict[str, ComponentDef] = field(default_factory=dict)
    geometry_sets: dict[str, GeometrySetDef] = field(default_factory=dict)
    load_definitions: dict[str, LoadDefinition] = field(default_factory=dict)
    boundary_definitions: dict[str, BoundaryDefinition] = field(default_factory=dict)
    default_component_id: str = "component:default"
    mesh_backend: str = "builtin"
    global_mesh_size: float = 1.2
    edge_mesh_sizes: dict[str, float] = field(default_factory=dict)
    region_mesh_algorithms: dict[str, str] = field(default_factory=dict)

    def ensure_default_component(self) -> ComponentDef:
        component = self.components.get(self.default_component_id)
        if component is None:
            component = ComponentDef(
                id=self.default_component_id,
                name="Default Component",
                display_color="#2563eb",
                visible=True,
            )
            self.components[self.default_component_id] = component
        return component

    def next_component_id(self) -> str:
        max_idx = 0
        for comp_id in self.components:
            token = comp_id.rsplit(":", 1)[-1]
            if token.isdigit():
                max_idx = max(max_idx, int(token))
        return f"component:{max_idx + 1}"

    def ensure_component(self, *, component_id: str | None = None, name: str | None = None) -> ComponentDef:
        if component_id is None:
            component_id = self.next_component_id()
        existing = self.components.get(component_id)
        if existing is not None:
            if name:
                existing.name = name
            return existing
        component = ComponentDef(
            id=component_id,
            name=name or component_id,
            display_color="#2563eb",
            visible=True,
        )
        self.components[component_id] = component
        return component

    def rebuild_component_regions(self) -> None:
        for component in self.components.values():
            component.region_ids.clear()
        for region_id, region in self.regions.items():
            component = self.components.get(region.component_id)
            if component is None:
                component = self.ensure_default_component()
                region.component_id = component.id
            component.region_ids.append(region_id)

    def clear_sketch_entities(self) -> None:
        self.loops = {
            loop_id: loop
            for loop_id, loop in self.loops.items()
            if loop.source != "sketch"
        }
        self.geometry_points = {
            point_id: point
            for point_id, point in self.geometry_points.items()
            if point.source != "sketch"
        }
        self.geometry_edges = {
            edge_id: edge
            for edge_id, edge in self.geometry_edges.items()
            if edge.source != "sketch"
        }
        self.regions = {
            region_id: region
            for region_id, region in self.regions.items()
            if region.source != "sketch"
        }
        self.rebuild_component_regions()
