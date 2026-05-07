from __future__ import annotations

import random
from dataclasses import replace
from math import sqrt

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


def _node_id(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix + 1


def _signed_double_area(a: Node, b: Node, c: Node) -> float:
    return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)


def _triangle_quality(a: Node, b: Node, c: Node) -> float:
    area = 0.5 * abs(_signed_double_area(a, b, c))
    l2_ab = (a.x - b.x) ** 2 + (a.y - b.y) ** 2
    l2_bc = (b.x - c.x) ** 2 + (b.y - c.y) ** 2
    l2_ca = (c.x - a.x) ** 2 + (c.y - a.y) ** 2
    denom = l2_ab + l2_bc + l2_ca
    if denom == 0.0:
        return 0.0
    return 4.0 * sqrt(3.0) * area / denom


def _ccw_triangle(n1: int, n2: int, n3: int, node_by_id: dict[int, Node]) -> list[int]:
    a = node_by_id[n1]
    b = node_by_id[n2]
    c = node_by_id[n3]
    if _signed_double_area(a, b, c) < 0.0:
        return [n1, n3, n2]
    return [n1, n2, n3]


def _build_randomized_t3_model(
    *,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    seed: int,
    jitter_fraction: float,
    materials: dict[int, Material],
    material_for_cell,
) -> Model:
    rng = random.Random(seed)
    dx = lx / nx
    dy = ly / ny

    nodes: list[Node] = []
    for iy in range(ny + 1):
        y = iy * dy
        for ix in range(nx + 1):
            x = ix * dx
            if 0 < ix < nx and 0 < iy < ny:
                x += (2.0 * rng.random() - 1.0) * jitter_fraction * dx
                y += (2.0 * rng.random() - 1.0) * jitter_fraction * dy
            nodes.append(Node(id=_node_id(ix, iy, nx), x=x, y=y))

    node_by_id = {node.id: node for node in nodes}
    elements: list[Element] = []
    eid = 1
    for iy in range(ny):
        for ix in range(nx):
            n00 = _node_id(ix, iy, nx)
            n10 = _node_id(ix + 1, iy, nx)
            n11 = _node_id(ix + 1, iy + 1, nx)
            n01 = _node_id(ix, iy + 1, nx)
            mid = material_for_cell(ix, iy)

            if rng.random() < 0.5:
                tri_a = _ccw_triangle(n00, n10, n11, node_by_id)
                tri_b = _ccw_triangle(n00, n11, n01, node_by_id)
            else:
                tri_a = _ccw_triangle(n00, n10, n01, node_by_id)
                tri_b = _ccw_triangle(n10, n11, n01, node_by_id)

            elements.append(Element(id=eid, type="T3", connectivity=tri_a, material_id=mid))
            eid += 1
            elements.append(Element(id=eid, type="T3", connectivity=tri_b, material_id=mid))
            eid += 1

    return Model(
        mesh=Mesh(nodes=nodes, elements=elements),
        materials=list(materials.values()),
    )


def _apply_mixed_boundary_and_loads(
    model: Model,
    *,
    lx: float,
    left_ux: float,
    top_left_uy: float,
) -> None:
    tol = 1e-12
    left_nodes = [node for node in model.mesh.nodes if abs(node.x) <= tol]
    right_nodes = [node for node in model.mesh.nodes if abs(node.x - lx) <= tol]
    if not left_nodes or not right_nodes:
        raise ValueError("mesh must contain both left and right boundary nodes")

    bottom_left = min(left_nodes, key=lambda node: node.y)
    top_left = max(left_nodes, key=lambda node: node.y)

    boundary_conditions: list[BoundaryCondition] = []
    for node in left_nodes:
        boundary_conditions.append(BoundaryCondition(node_id=node.id, dof="ux", value=left_ux))
    boundary_conditions.append(BoundaryCondition(node_id=bottom_left.id, dof="uy", value=0.0))
    boundary_conditions.append(BoundaryCondition(node_id=top_left.id, dof="uy", value=top_left_uy))
    model.boundary_conditions = boundary_conditions

    loads: list[Load] = []
    sorted_right = sorted(right_nodes, key=lambda node: node.y)
    for idx, node in enumerate(sorted_right):
        loads.append(Load(node_id=node.id, dof="fx", value=80.0 * (1.0 + 0.1 * idx)))
    loads.append(Load(node_id=sorted_right[-1].id, dof="fy", value=-150.0))
    loads.append(Load(node_id=sorted_right[len(sorted_right) // 2].id, dof="fy", value=40.0))
    model.loads = loads


def _solve_both_backends(model: Model):
    pytest.importorskip("fem_core", reason="fem_core extension is not built in this environment")
    py_result = solve_linear_static(model, backend="python")
    cpp_result = solve_linear_static(model, backend="cpp")
    return py_result, cpp_result


def _assert_backend_consistency(
    py_result,
    cpp_result,
    *,
    disp_rtol: float,
    disp_atol: float,
    reaction_rtol: float,
    reaction_atol: float,
    stress_rtol: float,
    stress_atol: float,
) -> None:
    np.testing.assert_allclose(
        cpp_result.displacements,
        py_result.displacements,
        rtol=disp_rtol,
        atol=disp_atol,
    )
    np.testing.assert_allclose(
        cpp_result.reactions,
        py_result.reactions,
        rtol=reaction_rtol,
        atol=reaction_atol,
    )

    assert cpp_result.summary.node_count == py_result.summary.node_count
    assert cpp_result.summary.element_count == py_result.summary.element_count
    assert cpp_result.summary.total_dof == py_result.summary.total_dof
    assert cpp_result.summary.max_displacement == pytest.approx(py_result.summary.max_displacement)

    py_elements = {item.element_id: item for item in py_result.element_results}
    cpp_elements = {item.element_id: item for item in cpp_result.element_results}
    assert py_elements.keys() == cpp_elements.keys()
    for element_id in py_elements:
        np.testing.assert_allclose(
            cpp_elements[element_id].strain,
            py_elements[element_id].strain,
            rtol=disp_rtol,
            atol=disp_atol,
        )
        np.testing.assert_allclose(
            cpp_elements[element_id].stress,
            py_elements[element_id].stress,
            rtol=stress_rtol,
            atol=stress_atol,
        )


def _triangle_quality_stats(model: Model) -> tuple[float, float]:
    node_by_id = {node.id: node for node in model.mesh.nodes}
    qualities = [
        _triangle_quality(
            node_by_id[element.connectivity[0]],
            node_by_id[element.connectivity[1]],
            node_by_id[element.connectivity[2]],
        )
        for element in model.mesh.elements
    ]
    return min(qualities), float(np.mean(qualities))


def test_randomized_nonstructured_mesh_backend_consistency() -> None:
    material = Material(id=1, young_modulus=150e9, poisson_ratio=0.29, plane_stress=True)
    model = _build_randomized_t3_model(
        nx=7,
        ny=5,
        lx=3.0,
        ly=2.0,
        seed=20260329,
        jitter_fraction=0.33,
        materials={1: material},
        material_for_cell=lambda _ix, _iy: 1,
    )
    _apply_mixed_boundary_and_loads(model, lx=3.0, left_ux=0.0, top_left_uy=1.5e-5)

    py_result, cpp_result = _solve_both_backends(model)
    _assert_backend_consistency(
        py_result,
        cpp_result,
        disp_rtol=2e-8,
        disp_atol=1e-11,
        reaction_rtol=2e-8,
        reaction_atol=5e-5,
        stress_rtol=2e-8,
        stress_atol=5e-4,
    )

    assert len(py_result.element_results) == len(model.mesh.elements)
    assert np.all(np.isfinite(py_result.displacements))
    assert np.all(np.isfinite(cpp_result.reactions))


def test_distorted_slender_mesh_and_high_condition_combo_backend_consistency() -> None:
    materials = {
        1: Material(id=1, young_modulus=220e9, poisson_ratio=0.495, plane_stress=False),
        2: Material(id=2, young_modulus=220e6, poisson_ratio=0.30, plane_stress=False),
    }
    model = _build_randomized_t3_model(
        nx=12,
        ny=2,
        lx=18.0,
        ly=0.30,
        seed=20260330,
        jitter_fraction=0.35,
        materials=materials,
        material_for_cell=lambda ix, _iy: 1 if ix % 2 == 0 else 2,
    )
    _apply_mixed_boundary_and_loads(model, lx=18.0, left_ux=2.0e-6, top_left_uy=8.0e-7)

    min_quality, mean_quality = _triangle_quality_stats(model)
    assert min_quality > 0.0
    assert min_quality < 0.20
    assert mean_quality < 0.55

    py_result, cpp_result = _solve_both_backends(model)
    _assert_backend_consistency(
        py_result,
        cpp_result,
        disp_rtol=2e-6,
        disp_atol=1e-10,
        reaction_rtol=5e-6,
        reaction_atol=5e-3,
        stress_rtol=5e-6,
        stress_atol=5e-2,
    )


@pytest.mark.parametrize("plane_stress", [True, False])
def test_randomized_mesh_plane_mode_consistency(plane_stress: bool) -> None:
    material = Material(
        id=1,
        young_modulus=185e9,
        poisson_ratio=0.31 if plane_stress else 0.30,
        plane_stress=plane_stress,
    )
    model = _build_randomized_t3_model(
        nx=6,
        ny=4,
        lx=2.5,
        ly=1.3,
        seed=20260331,
        jitter_fraction=0.25,
        materials={1: material},
        material_for_cell=lambda _ix, _iy: 1,
    )
    _apply_mixed_boundary_and_loads(model, lx=2.5, left_ux=0.0, top_left_uy=6.0e-6)

    py_result, cpp_result = _solve_both_backends(model)
    _assert_backend_consistency(
        py_result,
        cpp_result,
        disp_rtol=5e-8,
        disp_atol=1e-11,
        reaction_rtol=5e-8,
        reaction_atol=1e-4,
        stress_rtol=5e-8,
        stress_atol=1e-3,
    )


def test_randomized_mesh_stress_vs_strain_modes_remain_distinct_on_both_backends() -> None:
    base_material = Material(id=1, young_modulus=200e9, poisson_ratio=0.28, plane_stress=True)

    def build_model(material: Material) -> Model:
        model = _build_randomized_t3_model(
            nx=5,
            ny=4,
            lx=2.2,
            ly=1.6,
            seed=20260332,
            jitter_fraction=0.22,
            materials={1: material},
            material_for_cell=lambda _ix, _iy: 1,
        )
        _apply_mixed_boundary_and_loads(model, lx=2.2, left_ux=0.0, top_left_uy=4.0e-6)
        return model

    stress_model = build_model(replace(base_material, plane_stress=True))
    strain_model = build_model(replace(base_material, plane_stress=False))

    py_stress, cpp_stress = _solve_both_backends(stress_model)
    py_strain, cpp_strain = _solve_both_backends(strain_model)

    _assert_backend_consistency(
        py_stress,
        cpp_stress,
        disp_rtol=5e-8,
        disp_atol=1e-11,
        reaction_rtol=5e-8,
        reaction_atol=1e-4,
        stress_rtol=5e-8,
        stress_atol=1e-3,
    )
    _assert_backend_consistency(
        py_strain,
        cpp_strain,
        disp_rtol=5e-8,
        disp_atol=1e-11,
        reaction_rtol=5e-8,
        reaction_atol=1e-4,
        stress_rtol=5e-8,
        stress_atol=1e-3,
    )

    py_mode_diff = np.max(np.abs(py_stress.displacements - py_strain.displacements))
    cpp_mode_diff = np.max(np.abs(cpp_stress.displacements - cpp_strain.displacements))
    assert py_mode_diff > 1e-14
    assert cpp_mode_diff > 1e-14
