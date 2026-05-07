from __future__ import annotations

from statistics import mean

import numpy as np

from fem_ai_solver.fem.model import (
    BoundaryCondition,
    Element,
    Material,
    Mesh,
    Model,
    Node,
)
from fem_ai_solver.fem.services import solve_linear_static


def node_id(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix + 1


def build_multimaterial_patch_model() -> Model:
    nx, ny = 4, 2
    lx, ly = 4.0, 1.0

    nodes: list[Node] = []
    for iy in range(ny + 1):
        y = ly * iy / ny
        for ix in range(nx + 1):
            x = lx * ix / nx
            nodes.append(Node(id=node_id(ix, iy, nx), x=x, y=y))

    elements: list[Element] = []
    eid = 1
    for iy in range(ny):
        for ix in range(nx):
            n00 = node_id(ix, iy, nx)
            n10 = node_id(ix + 1, iy, nx)
            n11 = node_id(ix + 1, iy + 1, nx)
            n01 = node_id(ix, iy + 1, nx)
            material_id = 1 if ix < 2 else 2
            elements.append(Element(id=eid, type="T3", connectivity=[n00, n10, n11], material_id=material_id))
            eid += 1
            elements.append(Element(id=eid, type="T3", connectivity=[n00, n11, n01], material_id=material_id))
            eid += 1

    a, b, c = 8.0e-4, 1.0e-4, 2.0e-5
    d, e, f = -2.0e-4, 5.0e-4, 4.0e-5

    def ux(x: float, y: float) -> float:
        return a * x + b * y + c

    def uy(x: float, y: float) -> float:
        return d * x + e * y + f

    boundary_conditions: list[BoundaryCondition] = []
    for node in nodes:
        boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="ux", value=ux(node.x, node.y)))
        boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="uy", value=uy(node.x, node.y)))

    return Model(
        mesh=Mesh(nodes=nodes, elements=elements),
        materials=[
            Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True),
            Material(id=2, young_modulus=70e9, poisson_ratio=0.3, plane_stress=True),
        ],
        boundary_conditions=boundary_conditions,
    )


def summarize(tag: str, result, model: Model) -> None:
    print(f"[{tag}] max_u={result.summary.max_displacement:.6e}, dof={result.summary.total_dof}")
    stress_x = {1: [], 2: []}
    element_to_material = {element.id: element.material_id for element in model.mesh.elements}
    for item in result.element_results:
        stress_x[element_to_material[item.element_id]].append(item.stress[0])

    print(f"[{tag}] mean sigma_x material 1 = {mean(stress_x[1]):.6e}")
    print(f"[{tag}] mean sigma_x material 2 = {mean(stress_x[2]):.6e}")


if __name__ == "__main__":
    model = build_multimaterial_patch_model()

    result_python = solve_linear_static(model, backend="python")
    summarize("python", result_python, model)

    try:
        result_cpp = solve_linear_static(model, backend="cpp")
    except RuntimeError as exc:
        print(f"[cpp] unavailable: {exc}")
        raise SystemExit(0)

    summarize("cpp", result_cpp, model)

    displacement_diff = np.max(np.abs(result_python.displacements - result_cpp.displacements))
    reaction_diff = np.max(np.abs(result_python.reactions - result_cpp.reactions))

    print(f"[diff] max |u_py-u_cpp| = {displacement_diff:.6e}")
    print(f"[diff] max |r_py-r_cpp| = {reaction_diff:.6e}")
