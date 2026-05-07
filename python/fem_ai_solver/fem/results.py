from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class NodeDisplacement:
    node_id: int
    ux: float
    uy: float


@dataclass(slots=True)
class ElementStrainStress:
    element_id: int
    strain: tuple[float, float, float]
    stress: tuple[float, float, float]


@dataclass(slots=True)
class StaticSolveSummary:
    node_count: int
    element_count: int
    total_dof: int
    max_displacement: float


@dataclass(slots=True)
class StaticSolveResult:
    displacements: FloatArray
    reactions: FloatArray
    node_displacements: list[NodeDisplacement]
    element_results: list[ElementStrainStress]
    summary: StaticSolveSummary
