from __future__ import annotations

from fem_ai_solver.fem.services import solve_linear_static
from fem_ai_solver.preprocessing.demo_model import build_stage1_demo_model


def test_stage1_demo_model_is_solver_ready() -> None:
    model = build_stage1_demo_model()

    result = solve_linear_static(model, backend="python")

    assert result.summary.node_count == len(model.mesh.nodes)
    assert result.summary.element_count == len(model.mesh.elements)
    assert len(result.node_displacements) == len(model.mesh.nodes)


def test_model_from_t3_mesh_can_preserve_material_ids() -> None:
    from fem_ai_solver.fem.model import Element, Mesh, Node
    from fem_ai_solver.preprocessing.demo_model import model_from_t3_mesh

    mesh = Mesh(
        nodes=[
            Node(id=1, x=0.0, y=0.0),
            Node(id=2, x=1.0, y=0.0),
            Node(id=3, x=0.0, y=1.0),
        ],
        elements=[Element(id=1, type='T3', connectivity=[1, 2, 3], material_id=7)],
    )

    model = model_from_t3_mesh(mesh, preserve_element_material_ids=True)
    assert [item.material_id for item in model.mesh.elements] == [7]
    assert [item.id for item in model.materials] == [7]

