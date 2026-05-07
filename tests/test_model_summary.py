from fem_ai_solver.fem.model import Material, Mesh, Model, Node
from fem_ai_solver.fem.services import summarize_model


def test_summarize_model_counts_entities() -> None:
    model = Model(
        mesh=Mesh(nodes=[Node(id=1, x=0.0, y=0.0)]),
        materials=[Material(id=1, young_modulus=210e9, poisson_ratio=0.3)],
    )

    summary = summarize_model(model)

    assert summary["node_count"] == 1
    assert summary["material_count"] == 1
