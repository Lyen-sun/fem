from __future__ import annotations

import numpy as np
import pytest

from fem_ai_solver.fem.model import Material, Node
from fem_ai_solver.fem import services


def _load_fem_core():
    return pytest.importorskip(
        "fem_core",
        reason="fem_core extension is not built in this environment",
    )


def _make_material() -> Material:
    return Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)


def _make_nodes() -> list[Node]:
    return [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=0.0, y=1.0),
    ]


def _to_core_material(fem_core, material: Material):
    core_material = fem_core.Material()
    core_material.id = material.id
    core_material.young_modulus = material.young_modulus
    core_material.poisson_ratio = material.poisson_ratio
    core_material.plane_stress = material.plane_stress
    return core_material


def _to_core_nodes(fem_core, nodes: list[Node]):
    core_nodes = []
    for node in nodes:
        core_node = fem_core.Node()
        core_node.id = node.id
        core_node.x = node.x
        core_node.y = node.y
        core_nodes.append(core_node)
    return core_nodes


def test_cpp_elastic_matrix_matches_python_baseline() -> None:
    fem_core = _load_fem_core()
    material = _make_material()

    cpp = np.asarray(
        fem_core.elastic_matrix(_to_core_material(fem_core, material)),
        dtype=np.float64,
    ).reshape(3, 3)
    py = np.asarray(services._elastic_matrix_py(material), dtype=np.float64)

    np.testing.assert_allclose(cpp, py, rtol=1e-12, atol=1e-12)


def test_cpp_t3_area_matches_python_baseline() -> None:
    fem_core = _load_fem_core()
    nodes = _make_nodes()

    cpp = fem_core.t3_area(_to_core_nodes(fem_core, nodes))
    py = services.t3_area(nodes)

    assert cpp == pytest.approx(py)
    assert cpp == pytest.approx(0.5)


def test_cpp_t3_b_matrix_matches_python_baseline() -> None:
    fem_core = _load_fem_core()
    nodes = _make_nodes()

    cpp = np.asarray(
        fem_core.t3_b_matrix(_to_core_nodes(fem_core, nodes)),
        dtype=np.float64,
    ).reshape(3, 6)
    py = np.asarray(services._t3_b_matrix_py(nodes), dtype=np.float64)

    np.testing.assert_allclose(cpp, py, rtol=1e-12, atol=1e-12)


def test_cpp_t3_stiffness_matches_python_baseline() -> None:
    fem_core = _load_fem_core()
    nodes = _make_nodes()
    material = _make_material()

    cpp = np.asarray(
        fem_core.t3_stiffness_matrix(
            _to_core_nodes(fem_core, nodes),
            _to_core_material(fem_core, material),
            1.0,
        ),
        dtype=np.float64,
    ).reshape(6, 6)
    py = np.asarray(services._t3_stiffness_matrix_py(nodes, material), dtype=np.float64)

    np.testing.assert_allclose(cpp, py, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(cpp, cpp.T, rtol=1e-12, atol=1e-12)
