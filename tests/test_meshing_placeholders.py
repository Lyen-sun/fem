from __future__ import annotations

import pytest

from fem_ai_solver.mesh.adaptive.refinement import (
    AdaptiveRefinementInput,
    ElementIndicator,
    SizeField,
    refine_t3_mesh_by_size_field,
)
from fem_ai_solver.mesh.generation.constrained_delaunay import (
    BoundaryLoop,
    ConstrainedDelaunayInput,
    ConstrainedDelaunayResult,
    RegionMarker,
    generate_constrained_delaunay_t3,
)


def test_constrained_delaunay_returns_mesh_with_point_mapping() -> None:
    payload = ConstrainedDelaunayInput(
        points=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        segments=[(0, 1), (1, 2), (2, 0)],
        outer_loop=BoundaryLoop(point_ids=[0, 1, 2]),
        holes=[],
        regions=[RegionMarker(x=0.2, y=0.2, material_id=7)],
        embedded_point_ids=[0, 1, 2],
    )

    result = generate_constrained_delaunay_t3(payload)
    assert isinstance(result, ConstrainedDelaunayResult)
    assert len(result.mesh.nodes) == 3
    assert len(result.mesh.elements) >= 1
    assert result.input_point_to_node_id == {0: 1, 1: 2, 2: 3}
    assert all(element.material_id == 7 for element in result.mesh.elements)


def test_adaptive_refinement_placeholder_contract() -> None:
    from fem_ai_solver.fem.model import Mesh, Node

    payload = AdaptiveRefinementInput(
        mesh=Mesh(nodes=[Node(id=1, x=0.0, y=0.0)], elements=[]),
        size_field=SizeField(default_size=0.1),
        indicators=[ElementIndicator(element_id=1, value=0.2)],
    )

    with pytest.raises(NotImplementedError, match="Adaptive T3 remeshing"):
        refine_t3_mesh_by_size_field(payload)
