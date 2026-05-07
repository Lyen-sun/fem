from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ElementType = Literal["T3", "Q4", "T6"]


@dataclass(slots=True)
class Node:
    id: int
    x: float
    y: float


@dataclass(slots=True)
class Material:
    id: int
    young_modulus: float
    poisson_ratio: float
    plane_stress: bool = True


@dataclass(slots=True)
class Element:
    id: int
    type: ElementType
    connectivity: list[int]
    material_id: int


@dataclass(slots=True)
class BoundaryCondition:
    node_id: int
    dof: Literal["ux", "uy"]
    value: float = 0.0


@dataclass(slots=True)
class Load:
    node_id: int
    dof: Literal["fx", "fy"]
    value: float


@dataclass(slots=True)
class Mesh:
    nodes: list[Node] = field(default_factory=list)
    elements: list[Element] = field(default_factory=list)


@dataclass(slots=True)
class Model:
    mesh: Mesh
    materials: list[Material] = field(default_factory=list)
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)
