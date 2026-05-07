from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from fem_ai_solver.ai.geotech_assistant import parse_geotech_text_description
from fem_ai_solver.fem.model import Material
from fem_ai_solver.geotech.models import (
    GeotechLayer,
    GeotechTemplateInput,
    LayerBoundary,
    PointOfInterest,
)


def _parse_bool(text: str | None, default: bool = False) -> bool:
    if text is None:
        return default
    return text.strip().lower() in {"1", "true", "yes", "y", "on"}


def _sort_boundary_points(
    rows: list[dict[str, str]],
    layer_name: str,
    boundary_name: str,
) -> list[tuple[float, float]]:
    if not rows:
        raise ValueError(f"Layer {layer_name!r} is missing {boundary_name!r} boundary points.")

    if all("point_order" in row and row["point_order"].strip() for row in rows):
        rows = sorted(rows, key=lambda row: int(row["point_order"]))
    else:
        rows = sorted(rows, key=lambda row: float(row["x"]))

    points = [(float(row["x"]), float(row["y"])) for row in rows]
    if len(points) < 2:
        raise ValueError(
            f"Layer {layer_name!r} boundary {boundary_name!r} must contain at least 2 points."
        )
    return points


def load_geotech_layers_from_csv(
    path: str | Path,
    *,
    default_plane_stress: bool = False,
) -> list[GeotechLayer]:
    """Load geotechnical layered section definitions from CSV.

    Required columns:
    - layer
    - boundary   (top or bottom)
    - x
    - y

    Optional columns:
    - point_order
    - material_id
    - young_modulus
    - poisson_ratio
    - plane_stress
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValueError(f"CSV file does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no layer rows.")

    required = {"layer", "boundary", "x", "y"}
    missing = required.difference({name.strip() for name in reader.fieldnames})
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        layer = (row.get("layer") or "").strip()
        boundary = (row.get("boundary") or "").strip().lower()
        if not layer:
            raise ValueError("CSV row has empty 'layer' value.")
        if boundary not in {"top", "bottom"}:
            raise ValueError(
                f"CSV row has invalid boundary {boundary!r}. Expected 'top' or 'bottom'."
            )
        grouped[layer][boundary].append(row)

    layers: list[GeotechLayer] = []
    for index, (layer_name, by_boundary) in enumerate(grouped.items(), start=1):
        top_points = _sort_boundary_points(by_boundary["top"], layer_name, "top")
        bottom_points = _sort_boundary_points(by_boundary["bottom"], layer_name, "bottom")

        material_row = by_boundary["top"][0]
        material_id = int(material_row.get("material_id") or index)
        if material_id <= 0:
            raise ValueError(f"Layer {layer_name!r} has invalid material_id={material_id}.")

        if not material_row.get("young_modulus") or not material_row.get("poisson_ratio"):
            raise ValueError(
                f"Layer {layer_name!r} must provide young_modulus and poisson_ratio in CSV."
            )

        material = Material(
            id=material_id,
            young_modulus=float(material_row["young_modulus"]),
            poisson_ratio=float(material_row["poisson_ratio"]),
            plane_stress=_parse_bool(
                material_row.get("plane_stress"),
                default=default_plane_stress,
            ),
        )

        layers.append(
            GeotechLayer(
                name=layer_name,
                material=material,
                top_boundary=LayerBoundary(points=top_points),
                bottom_boundary=LayerBoundary(points=bottom_points),
            )
        )

    return layers


def build_template_input_from_text_description(
    text: str,
    *,
    backend: str = "python",
    thickness: float = 1.0,
    area_tolerance: float = 1e-14,
) -> GeotechTemplateInput:
    """Build workflow input from optional AI/text description parsing.

    Text parser output is mapped to plane-strain materials (`plane_stress=False`)
    by default because this module targets 2D geotechnical template problems.
    """
    intent = parse_geotech_text_description(text)
    layers: list[GeotechLayer] = []
    for index, layer in enumerate(intent.layers, start=1):
        layers.append(
            GeotechLayer(
                name=layer.name,
                material=Material(
                    id=index,
                    young_modulus=layer.young_modulus,
                    poisson_ratio=layer.poisson_ratio,
                    plane_stress=False,
                ),
                top_boundary=LayerBoundary(points=list(layer.top_points)),
                bottom_boundary=LayerBoundary(points=list(layer.bottom_points)),
            )
        )

    points = [
        PointOfInterest(name=point.name, x=point.x, y=point.y)
        for point in intent.points_of_interest
    ]

    return GeotechTemplateInput(
        layers=layers,
        mesh_target_size=float(intent.mesh_target_size),
        top_line_load=float(intent.top_line_load),
        points_of_interest=points,
        backend=backend,
        thickness=thickness,
        area_tolerance=area_tolerance,
    )

