from __future__ import annotations

from dataclasses import replace
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
from fem_ai_solver.fem.services import solve_linear_static


def _elastic_matrix_local(material: Material) -> np.ndarray:
    e = material.young_modulus
    nu = material.poisson_ratio

    if material.plane_stress:
        factor = e / (1.0 - nu * nu)
        return np.array(
            [
                [factor, factor * nu, 0.0],
                [factor * nu, factor, 0.0],
                [0.0, 0.0, factor * (1.0 - nu) * 0.5],
            ],
            dtype=np.float64,
        )

    factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return np.array(
        [
            [factor * (1.0 - nu), factor * nu, 0.0],
            [factor * nu, factor * (1.0 - nu), 0.0],
            [0.0, 0.0, factor * (1.0 - 2.0 * nu) * 0.5],
        ],
        dtype=np.float64,
    )


def _node_id(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix + 1


def _structured_t3_model(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    materials: dict[int, Material],
    material_for_cell,
) -> Model:
    nodes: list[Node] = []
    for iy in range(ny + 1):
        y = ly * iy / ny
        for ix in range(nx + 1):
            x = lx * ix / nx
            nodes.append(Node(id=_node_id(ix, iy, nx), x=x, y=y))

    elements: list[Element] = []
    eid = 1
    for iy in range(ny):
        for ix in range(nx):
            n00 = _node_id(ix, iy, nx)
            n10 = _node_id(ix + 1, iy, nx)
            n11 = _node_id(ix + 1, iy + 1, nx)
            n01 = _node_id(ix, iy + 1, nx)
            mid = material_for_cell(ix, iy)
            elements.append(Element(id=eid, type="T3", connectivity=[n00, n10, n11], material_id=mid))
            eid += 1
            elements.append(Element(id=eid, type="T3", connectivity=[n00, n11, n01], material_id=mid))
            eid += 1

    return Model(
        mesh=Mesh(nodes=nodes, elements=elements),
        materials=list(materials.values()),
    )


def _solve_both_backends(model: Model):
    pytest.importorskip("fem_core", reason="fem_core extension is not built in this environment")
    py_result = solve_linear_static(model, backend="python")
    cpp_result = solve_linear_static(model, backend="cpp")
    return py_result, cpp_result


def _assert_backend_consistency(py_result, cpp_result) -> None:
    np.testing.assert_allclose(cpp_result.displacements, py_result.displacements, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(cpp_result.reactions, py_result.reactions, rtol=1e-8, atol=1e-5)

    assert cpp_result.summary.node_count == py_result.summary.node_count
    assert cpp_result.summary.element_count == py_result.summary.element_count
    assert cpp_result.summary.total_dof == py_result.summary.total_dof
    assert cpp_result.summary.max_displacement == pytest.approx(py_result.summary.max_displacement)

    cpp_elem = {item.element_id: item for item in cpp_result.element_results}
    py_elem = {item.element_id: item for item in py_result.element_results}
    assert cpp_elem.keys() == py_elem.keys()
    for element_id in cpp_elem:
        np.testing.assert_allclose(cpp_elem[element_id].strain, py_elem[element_id].strain, rtol=1e-8, atol=1e-11)
        np.testing.assert_allclose(cpp_elem[element_id].stress, py_elem[element_id].stress, rtol=1e-8, atol=1e-5)


@pytest.mark.parametrize("plane_stress", [True, False])
def test_complex_patch_test_two_backend_for_stress_and_strain_modes(plane_stress: bool) -> None:
    material = Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=plane_stress)
    model = _structured_t3_model(
        nx=2,
        ny=2,
        lx=2.0,
        ly=1.0,
        materials={1: material},
        material_for_cell=lambda _ix, _iy: 1,
    )

    a, b, c = 1.2e-3, -2.5e-4, 8.0e-5
    d, e, f = 4.5e-4, 9.0e-4, -3.0e-5

    def ux(x: float, y: float) -> float:
        return a * x + b * y + c

    def uy(x: float, y: float) -> float:
        return d * x + e * y + f

    model.boundary_conditions = []
    for node in model.mesh.nodes:
        model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="ux", value=ux(node.x, node.y)))
        model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="uy", value=uy(node.x, node.y)))

    py_result, cpp_result = _solve_both_backends(model)
    _assert_backend_consistency(py_result, cpp_result)

    expected_strain = np.array([a, e, b + d], dtype=np.float64)
    expected_stress = _elastic_matrix_local(material) @ expected_strain

    for result in (py_result, cpp_result):
        for element_result in result.element_results:
            np.testing.assert_allclose(element_result.strain, expected_strain, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(element_result.stress, expected_stress, rtol=1e-9, atol=1e-6)


def test_larger_multi_element_case_backend_consistency() -> None:
    material = Material(id=1, young_modulus=70e9, poisson_ratio=0.28, plane_stress=True)
    model = _structured_t3_model(
        nx=4,
        ny=3,
        lx=2.0,
        ly=1.5,
        materials={1: material},
        material_for_cell=lambda _ix, _iy: 1,
    )

    model.boundary_conditions = []
    for node in model.mesh.nodes:
        if abs(node.x) < 1e-12:
            model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="ux", value=0.0))
            model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="uy", value=0.0))

    model.loads = []
    for node in model.mesh.nodes:
        if abs(node.x - 2.0) < 1e-12:
            model.loads.append(Load(node_id=node.id, dof="fx", value=125.0))
    model.loads.append(Load(node_id=_node_id(4, 3, 4), dof="fy", value=-200.0))

    py_result, cpp_result = _solve_both_backends(model)
    _assert_backend_consistency(py_result, cpp_result)

    assert py_result.summary.element_count == 24
    assert cpp_result.summary.total_dof == 40
    assert all(isfinite(value) for value in py_result.displacements)
    assert all(isfinite(value) for value in cpp_result.reactions)


def test_multi_material_partition_patch_case_backend_consistency() -> None:
    materials = {
        1: Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True),
        2: Material(id=2, young_modulus=70e9, poisson_ratio=0.3, plane_stress=True),
    }
    nx = 4
    ny = 2
    model = _structured_t3_model(
        nx=nx,
        ny=ny,
        lx=4.0,
        ly=1.0,
        materials=materials,
        material_for_cell=lambda ix, _iy: 1 if ix < 2 else 2,
    )

    a, b, c = 8.0e-4, 1.0e-4, 2.0e-5
    d, e, f = -2.0e-4, 5.0e-4, 4.0e-5

    def ux(x: float, y: float) -> float:
        return a * x + b * y + c

    def uy(x: float, y: float) -> float:
        return d * x + e * y + f

    model.boundary_conditions = []
    for node in model.mesh.nodes:
        model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="ux", value=ux(node.x, node.y)))
        model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="uy", value=uy(node.x, node.y)))

    py_result, cpp_result = _solve_both_backends(model)
    _assert_backend_consistency(py_result, cpp_result)

    expected_strain = np.array([a, e, b + d], dtype=np.float64)
    material_by_id = {material.id: material for material in model.materials}
    element_by_id = {element.id: element for element in model.mesh.elements}

    for result in (py_result, cpp_result):
        stress_x_by_material: dict[int, list[float]] = {1: [], 2: []}
        for element_result in result.element_results:
            element = element_by_id[element_result.element_id]
            expected_stress = _elastic_matrix_local(material_by_id[element.material_id]) @ expected_strain
            np.testing.assert_allclose(element_result.strain, expected_strain, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(element_result.stress, expected_stress, rtol=1e-9, atol=1e-6)
            stress_x_by_material[element.material_id].append(element_result.stress[0])

        assert stress_x_by_material[1]
        assert stress_x_by_material[2]
        assert np.mean(stress_x_by_material[1]) != pytest.approx(np.mean(stress_x_by_material[2]))


def test_nonzero_displacement_and_load_combination_backend_consistency() -> None:
    material = Material(id=1, young_modulus=120e9, poisson_ratio=0.29, plane_stress=True)
    model = _structured_t3_model(
        nx=2,
        ny=2,
        lx=1.0,
        ly=1.0,
        materials={1: material},
        material_for_cell=lambda _ix, _iy: 1,
    )

    model.boundary_conditions = []
    for node in model.mesh.nodes:
        if abs(node.x) < 1e-12:
            model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="ux", value=1.0e-4))
            model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="uy", value=0.0))
        elif abs(node.y) < 1e-12:
            model.boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="uy", value=0.0))

    model.loads = [
        Load(node_id=_node_id(2, 2, 2), dof="fx", value=300.0),
        Load(node_id=_node_id(2, 2, 2), dof="fy", value=-120.0),
        Load(node_id=_node_id(2, 1, 2), dof="fx", value=150.0),
    ]

    py_result, cpp_result = _solve_both_backends(model)
    _assert_backend_consistency(py_result, cpp_result)

    py_node = {item.node_id: item for item in py_result.node_displacements}
    cpp_node = {item.node_id: item for item in cpp_result.node_displacements}
    for node_id in (_node_id(0, 0, 2), _node_id(0, 1, 2), _node_id(0, 2, 2)):
        assert py_node[node_id].ux == pytest.approx(1.0e-4)
        assert cpp_node[node_id].ux == pytest.approx(1.0e-4)


def test_plane_stress_and_plane_strain_modes_both_backends_consistent() -> None:
    base_material = Material(id=1, young_modulus=200e9, poisson_ratio=0.27, plane_stress=True)

    def build_model(material: Material) -> Model:
        model = _structured_t3_model(
            nx=3,
            ny=2,
            lx=1.5,
            ly=1.0,
            materials={1: material},
            material_for_cell=lambda _ix, _iy: 1,
        )
        model.boundary_conditions = [
            BoundaryCondition(node_id=1, dof="ux", value=0.0),
            BoundaryCondition(node_id=1, dof="uy", value=0.0),
            BoundaryCondition(node_id=2, dof="uy", value=0.0),
            BoundaryCondition(node_id=5, dof="ux", value=0.0),
        ]
        model.loads = [
            Load(node_id=_node_id(3, 2, 3), dof="fx", value=500.0),
            Load(node_id=_node_id(3, 2, 3), dof="fy", value=-200.0),
        ]
        return model

    stress_model = build_model(replace(base_material, plane_stress=True))
    strain_model = build_model(replace(base_material, plane_stress=False))

    py_stress, cpp_stress = _solve_both_backends(stress_model)
    py_strain, cpp_strain = _solve_both_backends(strain_model)

    _assert_backend_consistency(py_stress, cpp_stress)
    _assert_backend_consistency(py_strain, cpp_strain)

    diff = np.max(np.abs(py_stress.displacements - py_strain.displacements))
    assert diff > 1e-14
