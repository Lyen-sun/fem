from __future__ import annotations

import numpy as np
import pytest

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


def _model_for_backend_tests() -> Model:
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=1.0, y=1.0),
        Node(id=4, x=0.0, y=1.0),
    ]
    elements = [
        Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=1),
        Element(id=2, type="T3", connectivity=[1, 3, 4], material_id=1),
    ]

    return Model(
        mesh=Mesh(nodes=nodes, elements=elements),
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


def test_backend_python_matches_default_behavior() -> None:
    model = _model_for_backend_tests()

    default_result = solve_linear_static(model)
    python_result = solve_linear_static(model, backend="python")

    np.testing.assert_allclose(default_result.displacements, python_result.displacements)
    np.testing.assert_allclose(default_result.reactions, python_result.reactions)

    assert default_result.summary == python_result.summary
    assert len(default_result.node_displacements) == len(python_result.node_displacements)
    assert len(default_result.element_results) == len(python_result.element_results)


def test_backend_cpp_matches_python_baseline() -> None:
    pytest.importorskip("fem_core", reason="fem_core extension is not built in this environment")
    model = _model_for_backend_tests()

    python_result = solve_linear_static(model, backend="python")
    cpp_result = solve_linear_static(model, backend="cpp")

    np.testing.assert_allclose(cpp_result.displacements, python_result.displacements, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cpp_result.reactions, python_result.reactions, rtol=1e-9, atol=1e-6)
    assert cpp_result.summary.node_count == python_result.summary.node_count
    assert cpp_result.summary.element_count == python_result.summary.element_count
    assert cpp_result.summary.total_dof == python_result.summary.total_dof
    assert cpp_result.summary.max_displacement == pytest.approx(python_result.summary.max_displacement)


def test_invalid_backend_raises_error() -> None:
    model = _model_for_backend_tests()

    with pytest.raises(ValueError, match="backend must be either 'python' or 'cpp'"):
        solve_linear_static(model, backend="fortran")
