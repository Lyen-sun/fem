from __future__ import annotations

import pytest

from fem_ai_solver.fem.model import BoundaryCondition, Element, Load, Material, Mesh, Model, Node
from fem_ai_solver.fem.services import solve_linear_static


def _multi_element_model() -> Model:
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


def test_smoke_default_python_backend_multi_element_model() -> None:
    result = solve_linear_static(_multi_element_model())

    assert result.summary.node_count == 4
    assert result.summary.element_count == 2
    assert result.summary.total_dof == 8
    assert len(result.element_results) == 2


def test_smoke_cpp_backend_multi_element_model() -> None:
    pytest.importorskip("fem_core", reason="fem_core extension is not built in this environment")
    result = solve_linear_static(_multi_element_model(), backend="cpp")

    assert result.summary.node_count == 4
    assert result.summary.element_count == 2
    assert result.summary.total_dof == 8
    assert len(result.element_results) == 2


def test_smoke_exception_model() -> None:
    model = _multi_element_model()
    model.loads = [Load(node_id=999, dof="fx", value=1.0)]

    with pytest.raises(ValueError, match="Load references unknown node id"):
        solve_linear_static(model)
