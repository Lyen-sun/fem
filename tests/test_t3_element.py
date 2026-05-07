from math import isfinite

import pytest

from fem_ai_solver.fem.model import Material, Node
from fem_ai_solver.fem.services import elastic_matrix, t3_area, t3_stiffness_matrix


@pytest.fixture
def sample_nodes() -> list[Node]:
    return [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=0.0, y=1.0),
    ]


@pytest.fixture
def sample_material() -> Material:
    return Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)


@pytest.mark.parametrize("plane_stress", [True, False])
def test_elastic_matrix_has_expected_shape(plane_stress: bool) -> None:
    material = Material(
        id=1,
        young_modulus=210e9,
        poisson_ratio=0.3,
        plane_stress=plane_stress,
    )

    d = elastic_matrix(material)

    assert len(d) == 3
    assert all(len(row) == 3 for row in d)


def test_t3_stiffness_matrix_has_expected_shape(sample_nodes: list[Node], sample_material: Material) -> None:
    ke = t3_stiffness_matrix(sample_nodes, sample_material)

    assert len(ke) == 6
    assert all(len(row) == 6 for row in ke)


def test_t3_stiffness_matrix_is_symmetric(sample_nodes: list[Node], sample_material: Material) -> None:
    ke = t3_stiffness_matrix(sample_nodes, sample_material)

    for i in range(6):
        for j in range(6):
            assert ke[i][j] == pytest.approx(ke[j][i])


def test_t3_positive_area_element_produces_finite_stiffness(
    sample_nodes: list[Node],
    sample_material: Material,
) -> None:
    area = t3_area(sample_nodes)
    ke = t3_stiffness_matrix(sample_nodes, sample_material)

    assert area > 0.0
    assert all(isfinite(value) for row in ke for value in row)
    assert ke[0][0] > 0.0
