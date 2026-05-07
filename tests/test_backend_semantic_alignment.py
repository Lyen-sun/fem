from __future__ import annotations

import pytest

from fem_ai_solver.fem.model import BoundaryCondition, Element, Load, Material, Mesh, Model, Node
from fem_ai_solver.fem.results import ElementStrainStress, NodeDisplacement, StaticSolveSummary
from fem_ai_solver.fem.services import solve_linear_static


def _valid_model() -> Model:
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


def _clockwise_model() -> Model:
    return Model(
        mesh=Mesh(
            nodes=[Node(id=1, x=0.0, y=0.0), Node(id=2, x=1.0, y=0.0), Node(id=3, x=0.0, y=1.0)],
            elements=[Element(id=1, type="T3", connectivity=[1, 3, 2], material_id=1)],
        ),
        materials=[Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)],
    )


def _degenerate_model() -> Model:
    return Model(
        mesh=Mesh(
            nodes=[Node(id=1, x=0.0, y=0.0), Node(id=2, x=1.0, y=0.0), Node(id=3, x=2.0, y=0.0)],
            elements=[Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=1)],
        ),
        materials=[Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)],
    )


def _underconstrained_model() -> Model:
    return Model(
        mesh=Mesh(
            nodes=[Node(id=1, x=0.0, y=0.0), Node(id=2, x=1.0, y=0.0), Node(id=3, x=0.0, y=1.0)],
            elements=[Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=1)],
        ),
        materials=[Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)],
        loads=[Load(node_id=2, dof="fx", value=100.0)],
    )


def _conflicting_bc_model() -> Model:
    model = _valid_model()
    model.boundary_conditions = [
        BoundaryCondition(node_id=1, dof="ux", value=0.0),
        BoundaryCondition(node_id=1, dof="ux", value=1.0),
    ]
    return model


def _unknown_load_node_model() -> Model:
    model = _valid_model()
    model.loads = [Load(node_id=999, dof="fx", value=1.0)]
    return model


def _unknown_bc_node_model() -> Model:
    model = _valid_model()
    model.boundary_conditions = [BoundaryCondition(node_id=999, dof="ux", value=0.0)]
    return model


@pytest.mark.parametrize(
    ("builder", "exc_type", "message_part"),
    [
        (_clockwise_model, ValueError, "clockwise node ordering"),
        (_degenerate_model, ValueError, "near-zero area"),
        (_underconstrained_model, RuntimeError, "insufficient displacement constraints"),
        (_conflicting_bc_model, ValueError, "Conflicting displacement constraints on global dof"),
        (_unknown_load_node_model, ValueError, "Load references unknown node id"),
        (_unknown_bc_node_model, ValueError, "Boundary condition references unknown node id"),
    ],
)
def test_backend_exception_semantics_alignment(builder, exc_type, message_part) -> None:
    pytest.importorskip("fem_core", reason="fem_core extension is not built in this environment")
    model = builder()

    with pytest.raises(exc_type) as py_exc:
        solve_linear_static(model, backend="python")

    with pytest.raises(exc_type) as cpp_exc:
        solve_linear_static(model, backend="cpp")

    assert message_part in str(py_exc.value)
    assert message_part in str(cpp_exc.value)


def test_backend_result_shape_and_type_alignment() -> None:
    pytest.importorskip("fem_core", reason="fem_core extension is not built in this environment")
    model = _valid_model()

    py_result = solve_linear_static(model, backend="python")
    cpp_result = solve_linear_static(model, backend="cpp")

    assert type(cpp_result.summary) is type(py_result.summary) is StaticSolveSummary
    assert all(isinstance(item, NodeDisplacement) for item in cpp_result.node_displacements)
    assert all(isinstance(item, ElementStrainStress) for item in cpp_result.element_results)

    assert cpp_result.displacements.shape == py_result.displacements.shape
    assert cpp_result.reactions.shape == py_result.reactions.shape
    assert len(cpp_result.node_displacements) == len(py_result.node_displacements)
    assert len(cpp_result.element_results) == len(py_result.element_results)
