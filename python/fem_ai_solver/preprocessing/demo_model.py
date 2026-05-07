from __future__ import annotations

from dataclasses import replace

from fem_ai_solver.fem.model import BoundaryCondition, Element, Load, Material, Mesh, Model, Node


def _node_id(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix + 1


def _structured_t3_mesh(nx: int, ny: int, lx: float, ly: float, material_id: int) -> Mesh:
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")

    nodes: list[Node] = []
    for iy in range(ny + 1):
        y = ly * iy / ny
        for ix in range(nx + 1):
            x = lx * ix / nx
            nodes.append(Node(id=_node_id(ix, iy, nx), x=x, y=y))

    elements: list[Element] = []
    element_id = 1
    for iy in range(ny):
        for ix in range(nx):
            n00 = _node_id(ix, iy, nx)
            n10 = _node_id(ix + 1, iy, nx)
            n11 = _node_id(ix + 1, iy + 1, nx)
            n01 = _node_id(ix, iy + 1, nx)

            elements.append(
                Element(
                    id=element_id,
                    type="T3",
                    connectivity=[n00, n10, n11],
                    material_id=material_id,
                )
            )
            element_id += 1
            elements.append(
                Element(
                    id=element_id,
                    type="T3",
                    connectivity=[n00, n11, n01],
                    material_id=material_id,
                )
            )
            element_id += 1

    return Mesh(nodes=nodes, elements=elements)


def model_from_t3_mesh(
    mesh: Mesh,
    material: Material | None = None,
    *,
    preserve_element_material_ids: bool = False,
) -> Model:
    """Create a solvable-ready model shell from a T3 mesh.

    By default all elements are assigned to one material. If
    `preserve_element_material_ids=True`, element material ids are preserved and
    one material object is generated per unique material id.
    """
    if not mesh.nodes:
        raise ValueError("mesh must contain at least one node")
    if not mesh.elements:
        raise ValueError("mesh must contain at least one element")

    base_material = material or Material(
        id=1,
        young_modulus=210e9,
        poisson_ratio=0.3,
        plane_stress=True,
    )

    if preserve_element_material_ids:
        unique_ids = sorted({element.material_id for element in mesh.elements})
        materials = [replace(base_material, id=material_id) for material_id in unique_ids]
        elements = list(mesh.elements)
    else:
        materials = [base_material]
        elements = [
            replace(element, material_id=base_material.id)
            for element in mesh.elements
        ]

    return Model(
        mesh=Mesh(nodes=list(mesh.nodes), elements=elements),
        materials=materials,
    )


def apply_left_clamp_right_nodal_force(
    model: Model,
    *,
    fx_total: float = 1_000.0,
    fy_top: float = -200.0,
) -> None:
    """Attach a deterministic default load case for quick UI demonstrations."""
    if not model.mesh.nodes:
        raise ValueError("model mesh has no nodes")

    min_x = min(node.x for node in model.mesh.nodes)
    max_x = max(node.x for node in model.mesh.nodes)
    tol = 1e-12

    left_nodes = [node for node in model.mesh.nodes if abs(node.x - min_x) <= tol]
    right_nodes = [node for node in model.mesh.nodes if abs(node.x - max_x) <= tol]

    if not left_nodes or not right_nodes:
        raise ValueError("model must include nodes on both left and right boundaries")

    model.boundary_conditions = []
    for node in left_nodes:
        model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="ux", value=0.0))
        model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="uy", value=0.0))

    fx_per_node = fx_total / len(right_nodes)
    top_right = max(right_nodes, key=lambda item: item.y)

    model.loads = []
    for node in right_nodes:
        model.loads.append(Load(node_id=node.id, dof="fx", value=fx_per_node))
    model.loads.append(Load(node_id=top_right.id, dof="fy", value=fy_top))


def build_stage1_demo_model() -> Model:
    """Build a minimal but non-trivial model for first-stage UI/solver demos."""
    material = Material(id=1, young_modulus=200e9, poisson_ratio=0.29, plane_stress=True)
    mesh = _structured_t3_mesh(nx=4, ny=2, lx=2.0, ly=1.0, material_id=material.id)
    model = model_from_t3_mesh(mesh, material)
    apply_left_clamp_right_nodal_force(model, fx_total=1_200.0, fy_top=-250.0)
    return model
