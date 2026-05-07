from __future__ import annotations

from dataclasses import dataclass

from fem_ai_solver.fem.model import Mesh


@dataclass(slots=True)
class SizeField:
    default_size: float


@dataclass(slots=True)
class ElementIndicator:
    element_id: int
    value: float


@dataclass(slots=True)
class AdaptiveRefinementInput:
    mesh: Mesh
    size_field: SizeField
    indicators: list[ElementIndicator]


def refine_t3_mesh_by_size_field(_input: AdaptiveRefinementInput) -> Mesh:
    """Placeholder for future size-field-driven adaptive T3 remeshing."""
    raise NotImplementedError(
        "Adaptive T3 remeshing is planned for a future stage. "
        "Current stage focuses on solver-stable import/preview/workflow."
    )
