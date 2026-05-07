from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class GeometryIntent:
    description: str
    dimension: Literal["2D"] = "2D"
    features: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SupportIntent:
    target: str
    constraint: str


@dataclass(slots=True)
class LoadIntent:
    target: str
    kind: str
    direction: str
    magnitude: str


@dataclass(slots=True)
class AnalysisIntent:
    raw_text: str
    geometry: GeometryIntent
    supports: list[SupportIntent] = field(default_factory=list)
    loads: list[LoadIntent] = field(default_factory=list)
    recommended_element: str = "T3"
    recommended_mesh_density: str = "medium"
