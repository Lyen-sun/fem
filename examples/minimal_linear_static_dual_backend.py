from __future__ import annotations

from fem_ai_solver.fem.model import (
    BoundaryCondition,
    Element,
    Load,
    Material,
    Mesh,
    Model,
    Node,
)
from fem_ai_solver.fem.services import solve_linear_static


def build_minimal_model() -> Model:
    return Model(
        mesh=Mesh(
            nodes=[
                Node(id=1, x=0.0, y=0.0),
                Node(id=2, x=1.0, y=0.0),
                Node(id=3, x=1.0, y=1.0),
                Node(id=4, x=0.0, y=1.0),
            ],
            elements=[
                Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=1),
                Element(id=2, type="T3", connectivity=[1, 3, 4], material_id=1),
            ],
        ),
        materials=[Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)],
        boundary_conditions=[
            BoundaryCondition(node_id=1, dof="ux", value=0.0),
            BoundaryCondition(node_id=1, dof="uy", value=0.0),
            BoundaryCondition(node_id=2, dof="uy", value=0.0),
            BoundaryCondition(node_id=4, dof="ux", value=0.0),
        ],
        loads=[
            Load(node_id=3, dof="fx", value=1000.0),
            Load(node_id=3, dof="fy", value=-500.0),
        ],
    )


def print_result(tag: str, result) -> None:
    print(f"[{tag}] total_dof={result.summary.total_dof}, max_u={result.summary.max_displacement:.6e}")
    print(f"[{tag}] u={result.displacements}")
    print(f"[{tag}] r={result.reactions}")


if __name__ == "__main__":
    model = build_minimal_model()

    result_python = solve_linear_static(model, backend="python")
    print_result("python", result_python)

    try:
        result_cpp = solve_linear_static(model, backend="cpp")
    except RuntimeError as exc:
        print(f"[cpp] unavailable: {exc}")
    else:
        print_result("cpp", result_cpp)

    invalid_model = build_minimal_model()
    invalid_model.loads = [Load(node_id=999, dof="fx", value=1.0)]
    try:
        solve_linear_static(invalid_model)
    except ValueError as exc:
        print(f"[invalid-model] expected error: {exc}")
