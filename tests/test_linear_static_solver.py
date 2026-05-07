from __future__ import annotations

from math import isfinite

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
from fem_ai_solver.fem.services import elastic_matrix, solve_linear_static


def _material() -> Material:
    return Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)


def _two_element_square_model() -> Model:
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
    return Model(mesh=Mesh(nodes=nodes, elements=elements), materials=[_material()])


def _single_t3_model(connectivity: list[int]) -> Model:
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=0.0, y=1.0),
    ]
    element = Element(id=1, type="T3", connectivity=connectivity, material_id=1)
    return Model(mesh=Mesh(nodes=nodes, elements=[element]), materials=[_material()])


def _solvable_model() -> Model:
    model = _two_element_square_model()
    model.loads = [
        Load(node_id=3, dof="fx", value=1000.0),
        Load(node_id=3, dof="fy", value=-500.0),
    ]
    model.boundary_conditions = [
        BoundaryCondition(node_id=1, dof="ux", value=0.0),
        BoundaryCondition(node_id=1, dof="uy", value=0.0),
        BoundaryCondition(node_id=2, dof="uy", value=0.0),
        BoundaryCondition(node_id=4, dof="ux", value=0.0),
    ]
    return model


def test_solve_linear_static_minimal_system() -> None:
    model = _solvable_model()

    result = solve_linear_static(model)

    assert result.summary.node_count == 4
    assert result.summary.element_count == 2
    assert result.summary.total_dof == 8

    assert result.displacements.shape == (8,)
    assert result.reactions.shape == (8,)
    assert np.all(np.isfinite(result.displacements))
    assert np.all(np.isfinite(result.reactions))

    # Node index mapping follows mesh order: node 1->0, node 2->1, node 4->3.
    assert result.displacements[0] == pytest.approx(0.0)
    assert result.displacements[1] == pytest.approx(0.0)
    assert result.displacements[3] == pytest.approx(0.0)
    assert result.displacements[6] == pytest.approx(0.0)


def test_element_strain_stress_recovery_outputs_finite_vectors() -> None:
    model = _solvable_model()

    result = solve_linear_static(model)

    assert len(result.element_results) == 2
    for element_result in result.element_results:
        assert len(element_result.strain) == 3
        assert len(element_result.stress) == 3
        assert all(isfinite(value) for value in element_result.strain)
        assert all(isfinite(value) for value in element_result.stress)


def test_clockwise_connectivity_raises_error() -> None:
    model = _single_t3_model(connectivity=[1, 3, 2])

    with pytest.raises(ValueError, match="clockwise"):
        solve_linear_static(model)


def test_zero_area_element_raises_error() -> None:
    model = _single_t3_model(connectivity=[1, 2, 3])
    model.mesh.nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=2.0, y=0.0),
    ]

    with pytest.raises(ValueError, match="near-zero area"):
        solve_linear_static(model)


def test_unknown_node_reference_in_element_raises_error() -> None:
    model = _single_t3_model(connectivity=[1, 2, 99])

    with pytest.raises(ValueError, match="unknown node id"):
        solve_linear_static(model)


def test_unknown_node_reference_in_load_raises_error() -> None:
    model = _single_t3_model(connectivity=[1, 2, 3])
    model.loads = [Load(node_id=99, dof="fx", value=1.0)]

    with pytest.raises(ValueError, match="Load references unknown node id"):
        solve_linear_static(model)


def test_unknown_node_reference_in_boundary_condition_raises_error() -> None:
    model = _single_t3_model(connectivity=[1, 2, 3])
    model.boundary_conditions = [BoundaryCondition(node_id=99, dof="ux", value=0.0)]

    with pytest.raises(ValueError, match="Boundary condition references unknown node id"):
        solve_linear_static(model)


def test_invalid_material_parameter_raises_error() -> None:
    model = _single_t3_model(connectivity=[1, 2, 3])
    model.materials[0].young_modulus = 0.0

    with pytest.raises(ValueError, match="young_modulus"):
        solve_linear_static(model)


def test_insufficient_constraints_raises_solver_error() -> None:
    model = _single_t3_model(connectivity=[1, 2, 3])
    model.loads = [Load(node_id=2, dof="fx", value=100.0)]

    with pytest.raises(RuntimeError, match="insufficient displacement constraints"):
        solve_linear_static(model)


def test_patch_test_constant_strain_field_on_two_elements() -> None:
    model = _two_element_square_model()

    # Prescribed linear displacement field:
    # u(x, y) = a*x + b*y + c
    # v(x, y) = d*x + e*y + f
    a, b, c = 1.0e-3, 2.0e-4, 5.0e-5
    d, e, f = -3.0e-4, 8.0e-4, -1.0e-5

    def u_value(x: float, y: float) -> float:
        return a * x + b * y + c

    def v_value(x: float, y: float) -> float:
        return d * x + e * y + f

    model.boundary_conditions = []
    for node in model.mesh.nodes:
        model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="ux", value=u_value(node.x, node.y)))
        model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="uy", value=v_value(node.x, node.y)))

    result = solve_linear_static(model)

    expected_strain = np.array([a, e, b + d], dtype=np.float64)
    expected_stress = np.asarray(elastic_matrix(model.materials[0]), dtype=np.float64) @ expected_strain

    for node_index, node in enumerate(model.mesh.nodes):
        np.testing.assert_allclose(result.displacements[2 * node_index], u_value(node.x, node.y), atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(result.displacements[2 * node_index + 1], v_value(node.x, node.y), atol=1e-12, rtol=1e-10)

    assert len(result.element_results) == 2
    for element_result in result.element_results:
        np.testing.assert_allclose(element_result.strain, expected_strain, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(element_result.stress, expected_stress, atol=1e-6, rtol=1e-10)

def test_empty_model_raises_error() -> None:
    model = Model(mesh=Mesh())

    with pytest.raises(ValueError, match="at least one node"):
        solve_linear_static(model)


def test_missing_material_raises_error() -> None:
    model = _single_t3_model(connectivity=[1, 2, 3])
    model.materials = []

    with pytest.raises(ValueError, match="at least one material"):
        solve_linear_static(model)
