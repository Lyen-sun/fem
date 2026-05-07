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
from fem_ai_solver.fem.services import solve_linear_static as solve_linear_static_py


def _load_fem_core():
    return pytest.importorskip(
        "fem_core",
        reason="fem_core extension is not built in this environment",
    )


def _model_for_parity() -> Model:
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
    materials = [Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)]
    boundary_conditions = [
        BoundaryCondition(node_id=1, dof="ux", value=0.0),
        BoundaryCondition(node_id=1, dof="uy", value=0.0),
        BoundaryCondition(node_id=2, dof="uy", value=0.0),
        BoundaryCondition(node_id=4, dof="ux", value=0.0),
    ]
    loads = [
        Load(node_id=3, dof="fx", value=1000.0),
        Load(node_id=3, dof="fy", value=-500.0),
    ]
    return Model(
        mesh=Mesh(nodes=nodes, elements=elements),
        materials=materials,
        boundary_conditions=boundary_conditions,
        loads=loads,
    )


def _to_core_nodes(fem_core, nodes: list[Node]):
    values = []
    for node in nodes:
        value = fem_core.Node()
        value.id = node.id
        value.x = node.x
        value.y = node.y
        values.append(value)
    return values


def _to_core_elements(fem_core, elements: list[Element]):
    values = []
    for element in elements:
        value = fem_core.Element()
        value.id = element.id
        value.type = element.type
        value.connectivity = element.connectivity
        value.material_id = element.material_id
        values.append(value)
    return values


def _to_core_materials(fem_core, materials: list[Material]):
    values = []
    for material in materials:
        value = fem_core.Material()
        value.id = material.id
        value.young_modulus = material.young_modulus
        value.poisson_ratio = material.poisson_ratio
        value.plane_stress = material.plane_stress
        values.append(value)
    return values


def _to_core_bcs(fem_core, bcs: list[BoundaryCondition]):
    values = []
    for bc in bcs:
        value = fem_core.BoundaryCondition()
        value.node_id = bc.node_id
        value.dof = bc.dof
        value.value = bc.value
        values.append(value)
    return values


def _to_core_loads(fem_core, loads: list[Load]):
    values = []
    for load in loads:
        value = fem_core.Load()
        value.node_id = load.node_id
        value.dof = load.dof
        value.value = load.value
        values.append(value)
    return values


def test_cpp_linear_static_matches_python_baseline() -> None:
    fem_core = _load_fem_core()
    model = _model_for_parity()

    py_result = solve_linear_static_py(model)
    cpp_result = fem_core.solve_linear_static(
        _to_core_nodes(fem_core, model.mesh.nodes),
        _to_core_elements(fem_core, model.mesh.elements),
        _to_core_materials(fem_core, model.materials),
        _to_core_bcs(fem_core, model.boundary_conditions),
        _to_core_loads(fem_core, model.loads),
        1.0,
        1e-14,
    )

    cpp_u = np.asarray(cpp_result.displacements, dtype=np.float64)
    cpp_r = np.asarray(cpp_result.reactions, dtype=np.float64)

    np.testing.assert_allclose(cpp_u, py_result.displacements, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cpp_r, py_result.reactions, rtol=1e-9, atol=1e-6)

    assert cpp_result.summary.node_count == py_result.summary.node_count
    assert cpp_result.summary.element_count == py_result.summary.element_count
    assert cpp_result.summary.total_dof == py_result.summary.total_dof
    assert cpp_result.summary.max_displacement == pytest.approx(py_result.summary.max_displacement)

    cpp_node_by_id = {item.node_id: item for item in cpp_result.node_displacements}
    py_node_by_id = {item.node_id: item for item in py_result.node_displacements}
    assert cpp_node_by_id.keys() == py_node_by_id.keys()
    for node_id in sorted(py_node_by_id):
        assert cpp_node_by_id[node_id].ux == pytest.approx(py_node_by_id[node_id].ux)
        assert cpp_node_by_id[node_id].uy == pytest.approx(py_node_by_id[node_id].uy)

    cpp_by_id = {item.element_id: item for item in cpp_result.element_results}
    py_by_id = {item.element_id: item for item in py_result.element_results}
    assert cpp_by_id.keys() == py_by_id.keys()

    for element_id in sorted(py_by_id):
        np.testing.assert_allclose(
            np.asarray(cpp_by_id[element_id].strain, dtype=np.float64),
            np.asarray(py_by_id[element_id].strain, dtype=np.float64),
            rtol=1e-9,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(cpp_by_id[element_id].stress, dtype=np.float64),
            np.asarray(py_by_id[element_id].stress, dtype=np.float64),
            rtol=1e-9,
            atol=1e-5,
        )


def test_cpp_linear_static_unknown_load_node_raises() -> None:
    fem_core = _load_fem_core()
    model = _model_for_parity()
    model.loads = [Load(node_id=999, dof="fx", value=1.0)]

    with pytest.raises(ValueError, match="Load references unknown node id"):
        fem_core.solve_linear_static(
            _to_core_nodes(fem_core, model.mesh.nodes),
            _to_core_elements(fem_core, model.mesh.elements),
            _to_core_materials(fem_core, model.materials),
            _to_core_bcs(fem_core, model.boundary_conditions),
            _to_core_loads(fem_core, model.loads),
            1.0,
            1e-14,
        )
