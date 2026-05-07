from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from fem_ai_solver.fem.model import BoundaryCondition, Element, Load, Material, Mesh, Model, Node
from fem_ai_solver.geotech.models import GeotechLayer, GeotechTemplateInput


@dataclass(slots=True)
class LayeredMeshBuild:
    """Generated FEM model plus boundary-node metadata for template loading."""

    model: Model
    top_surface_node_ids: list[int]
    min_x_node_ids: list[int]
    max_x_node_ids: list[int]
    min_y_node_ids: list[int]


def _require_finite_points(points: list[tuple[float, float]], label: str) -> None:
    for x, y in points:
        if not (isfinite(x) and isfinite(y)):
            raise ValueError(f"{label} contains non-finite point ({x}, {y}).")


def _normalize_boundary(points: list[tuple[float, float]], label: str) -> list[tuple[float, float]]:
    """Validate and normalize one y(x) boundary polyline."""
    if len(points) < 2:
        raise ValueError(f"{label} must contain at least 2 points.")
    _require_finite_points(points, label)

    sorted_points = sorted(points, key=lambda item: item[0])
    xs = [item[0] for item in sorted_points]
    for i in range(1, len(xs)):
        if abs(xs[i] - xs[i - 1]) <= 1e-12:
            raise ValueError(
                f"{label} has repeated x={xs[i]} values. "
                "Boundary points must be strictly increasing in x."
            )
    return sorted_points


def _interp_y(points: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear y(x) interpolation on normalized points."""
    if x < points[0][0] - 1e-12 or x > points[-1][0] + 1e-12:
        raise ValueError(
            f"x={x} is outside boundary range [{points[0][0]}, {points[-1][0]}]."
        )

    if abs(x - points[0][0]) <= 1e-12:
        return points[0][1]
    if abs(x - points[-1][0]) <= 1e-12:
        return points[-1][1]

    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if x0 <= x <= x1:
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    raise RuntimeError("Interpolation failed due to unexpected boundary indexing.")


def _canonical_coord(x: float, y: float) -> tuple[float, float]:
    # Coordinate snapping avoids duplicate nodes on shared interfaces.
    return (round(float(x), 12), round(float(y), 12))


def _signed_double_area(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])


def _validate_layers(layers: list[GeotechLayer], x_samples: np.ndarray) -> list[tuple[GeotechLayer, float]]:
    """Check layer validity, non-overlap, and ordering at sampled x stations."""
    if not layers:
        raise ValueError("At least one geotechnical layer is required.")

    enriched: list[tuple[GeotechLayer, float]] = []
    for layer in layers:
        top = _normalize_boundary(layer.top_boundary.points, f"{layer.name} top boundary")
        bottom = _normalize_boundary(layer.bottom_boundary.points, f"{layer.name} bottom boundary")
        if abs(top[0][0] - bottom[0][0]) > 1e-12 or abs(top[-1][0] - bottom[-1][0]) > 1e-12:
            raise ValueError(
                f"Layer {layer.name!r} top/bottom x-ranges must match for template meshing."
            )
        layer.top_boundary.points = top
        layer.bottom_boundary.points = bottom
        mid_x = 0.5 * (top[0][0] + top[-1][0])
        mean_mid_y = 0.5 * (_interp_y(top, mid_x) + _interp_y(bottom, mid_x))
        enriched.append((layer, mean_mid_y))

    # Sort layers from top to bottom by representative elevation.
    enriched.sort(key=lambda item: item[1], reverse=True)

    for layer, _ in enriched:
        for x in x_samples:
            top_y = _interp_y(layer.top_boundary.points, float(x))
            bottom_y = _interp_y(layer.bottom_boundary.points, float(x))
            if top_y <= bottom_y + 1e-12:
                raise ValueError(
                    f"Layer {layer.name!r} has non-positive thickness at x={x:.6g} "
                    f"(top={top_y}, bottom={bottom_y})."
                )

    # Ensure adjacent layers do not overlap after ordering.
    for i in range(len(enriched) - 1):
        upper = enriched[i][0]
        lower = enriched[i + 1][0]
        for x in x_samples:
            upper_bottom = _interp_y(upper.bottom_boundary.points, float(x))
            lower_top = _interp_y(lower.top_boundary.points, float(x))
            if lower_top > upper_bottom + 1e-10:
                raise ValueError(
                    f"Layer overlap detected between {upper.name!r} and {lower.name!r} "
                    f"at x={x:.6g}: lower top {lower_top} > upper bottom {upper_bottom}."
                )

    return enriched


def _build_x_samples(layers: list[GeotechLayer], mesh_target_size: float) -> np.ndarray:
    if mesh_target_size <= 0.0:
        raise ValueError("mesh_target_size must be positive.")

    all_x = [x for layer in layers for (x, _y) in layer.top_boundary.points + layer.bottom_boundary.points]
    min_x = min(all_x)
    max_x = max(all_x)
    if max_x <= min_x:
        raise ValueError("Invalid layer x-range. max_x must be greater than min_x.")

    span = max_x - min_x
    # At least 4 strips for stable template cases, then adapt by target size.
    nx = max(4, int(np.ceil(span / mesh_target_size)))
    return np.linspace(min_x, max_x, nx + 1, dtype=np.float64)


def _apply_template_boundary_and_loads(
    model: Model,
    *,
    top_surface_node_ids: list[int],
    min_x_node_ids: list[int],
    max_x_node_ids: list[int],
    min_y_node_ids: list[int],
    template: GeotechTemplateInput,
) -> None:
    """Apply geotechnical template boundary/load defaults."""
    bc_map: dict[tuple[int, str], float] = {}

    if template.fix_bottom_ux:
        for node_id in min_y_node_ids:
            bc_map[(node_id, "ux")] = 0.0
    if template.fix_bottom_uy:
        for node_id in min_y_node_ids:
            bc_map[(node_id, "uy")] = 0.0
    if template.constrain_lateral_ux:
        for node_id in min_x_node_ids + max_x_node_ids:
            bc_map[(node_id, "ux")] = 0.0

    model.boundary_conditions = [
        BoundaryCondition(node_id=node_id, dof=dof, value=value)
        for (node_id, dof), value in sorted(bc_map.items())
    ]

    if abs(template.top_line_load) <= 1e-14:
        model.loads = []
        return

    node_by_id = {node.id: node for node in model.mesh.nodes}
    top_nodes = sorted(top_surface_node_ids, key=lambda node_id: node_by_id[node_id].x)

    nodal_fy: dict[int, float] = {node_id: 0.0 for node_id in top_nodes}
    for i in range(len(top_nodes) - 1):
        n1 = node_by_id[top_nodes[i]]
        n2 = node_by_id[top_nodes[i + 1]]
        edge_length = float(np.hypot(n2.x - n1.x, n2.y - n1.y))
        # Consistent nodal load for a uniform line traction on a 2-node segment.
        segment_force = template.top_line_load * edge_length
        nodal_fy[n1.id] += -0.5 * segment_force
        nodal_fy[n2.id] += -0.5 * segment_force

    model.loads = [
        Load(node_id=node_id, dof="fy", value=value)
        for node_id, value in sorted(nodal_fy.items())
        if abs(value) > 1e-14
    ]


def build_layered_template_model(template: GeotechTemplateInput) -> LayeredMeshBuild:
    """Generate a T3 model from layered boundaries and apply template BC/loads."""
    if not template.layers:
        raise ValueError("template.layers cannot be empty.")

    x_samples = _build_x_samples(template.layers, template.mesh_target_size)
    enriched_layers = _validate_layers(template.layers, x_samples)

    node_id_by_coord: dict[tuple[float, float], int] = {}
    nodes: list[Node] = []

    def ensure_node(x: float, y: float) -> int:
        key = _canonical_coord(x, y)
        node_id = node_id_by_coord.get(key)
        if node_id is not None:
            return node_id
        node_id = len(nodes) + 1
        node_id_by_coord[key] = node_id
        nodes.append(Node(id=node_id, x=key[0], y=key[1]))
        return node_id

    elements: list[Element] = []
    top_surface_node_ids: set[int] = set()
    element_id = 1

    for layer_index, (layer, _) in enumerate(enriched_layers):
        material_id = layer.material.id
        for i in range(len(x_samples) - 1):
            x0 = float(x_samples[i])
            x1 = float(x_samples[i + 1])

            t0 = _interp_y(layer.top_boundary.points, x0)
            t1 = _interp_y(layer.top_boundary.points, x1)
            b0 = _interp_y(layer.bottom_boundary.points, x0)
            b1 = _interp_y(layer.bottom_boundary.points, x1)

            n_bl = ensure_node(x0, b0)
            n_br = ensure_node(x1, b1)
            n_tr = ensure_node(x1, t1)
            n_tl = ensure_node(x0, t0)

            # Keep element orientation CCW and explicitly reject inverted pieces.
            p_bl = (x0, b0)
            p_br = (x1, b1)
            p_tr = (x1, t1)
            p_tl = (x0, t0)

            if _signed_double_area(p_bl, p_br, p_tr) <= 0.0:
                raise ValueError(
                    f"Inverted triangle detected while meshing layer {layer.name!r} (cell {i})."
                )
            if _signed_double_area(p_bl, p_tr, p_tl) <= 0.0:
                raise ValueError(
                    f"Inverted triangle detected while meshing layer {layer.name!r} (cell {i}, split 2)."
                )

            elements.append(
                Element(
                    id=element_id,
                    type="T3",
                    connectivity=[n_bl, n_br, n_tr],
                    material_id=material_id,
                )
            )
            element_id += 1
            elements.append(
                Element(
                    id=element_id,
                    type="T3",
                    connectivity=[n_bl, n_tr, n_tl],
                    material_id=material_id,
                )
            )
            element_id += 1

            if layer_index == 0:
                top_surface_node_ids.update((n_tl, n_tr))

    nodes = sorted(nodes, key=lambda node: node.id)
    elements = sorted(elements, key=lambda element: element.id)
    material_by_id: dict[int, Material] = {}
    materials = []
    for layer, _ in enriched_layers:
        existing = material_by_id.get(layer.material.id)
        if existing is not None:
            if (
                layer.material.young_modulus != existing.young_modulus
                or layer.material.poisson_ratio != existing.poisson_ratio
                or layer.material.plane_stress != existing.plane_stress
            ):
                raise ValueError(
                    f"Material id {layer.material.id} is reused with inconsistent parameters "
                    f"in layer {layer.name!r}."
                )
            continue
        material_by_id[layer.material.id] = layer.material
        materials.append(layer.material)

    model = Model(mesh=Mesh(nodes=nodes, elements=elements), materials=materials)

    x_values = np.asarray([node.x for node in nodes], dtype=np.float64)
    y_values = np.asarray([node.y for node in nodes], dtype=np.float64)
    min_x = float(np.min(x_values))
    max_x = float(np.max(x_values))
    min_y = float(np.min(y_values))

    tol = 1e-10
    min_x_node_ids = [node.id for node in nodes if abs(node.x - min_x) <= tol]
    max_x_node_ids = [node.id for node in nodes if abs(node.x - max_x) <= tol]
    min_y_node_ids = [node.id for node in nodes if abs(node.y - min_y) <= tol]

    _apply_template_boundary_and_loads(
        model,
        top_surface_node_ids=sorted(top_surface_node_ids),
        min_x_node_ids=min_x_node_ids,
        max_x_node_ids=max_x_node_ids,
        min_y_node_ids=min_y_node_ids,
        template=template,
    )

    return LayeredMeshBuild(
        model=model,
        top_surface_node_ids=sorted(top_surface_node_ids),
        min_x_node_ids=min_x_node_ids,
        max_x_node_ids=max_x_node_ids,
        min_y_node_ids=min_y_node_ids,
    )
