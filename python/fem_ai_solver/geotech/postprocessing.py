from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fem_ai_solver.fem.model import Model
from fem_ai_solver.fem.results import StaticSolveResult
from fem_ai_solver.geotech.models import GeotechOutputFiles, GeotechPointResult, PointOfInterest


def _color_for_value(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    """Map scalar to an engineering-friendly blue->red RGB color."""
    if vmax <= vmin:
        return (52, 152, 219)
    ratio = (value - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))

    anchors: list[tuple[float, tuple[int, int, int]]] = [
        (0.00, (33, 102, 172)),
        (0.25, (67, 162, 202)),
        (0.50, (65, 171, 93)),
        (0.75, (241, 196, 15)),
        (1.00, (192, 57, 43)),
    ]

    for i in range(len(anchors) - 1):
        left_pos, left_rgb = anchors[i]
        right_pos, right_rgb = anchors[i + 1]
        if left_pos <= ratio <= right_pos:
            local = (ratio - left_pos) / (right_pos - left_pos)
            r = int(left_rgb[0] + (right_rgb[0] - left_rgb[0]) * local)
            g = int(left_rgb[1] + (right_rgb[1] - left_rgb[1]) * local)
            b = int(left_rgb[2] + (right_rgb[2] - left_rgb[2]) * local)
            return (r, g, b)
    return anchors[-1][1]


def compute_von_mises_by_element(result: StaticSolveResult) -> dict[int, float]:
    """Compute 2D Von Mises stress from (sx, sy, txy)."""
    values: dict[int, float] = {}
    for item in result.element_results:
        sx, sy, txy = item.stress
        vm = float(np.sqrt(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy))
        values[item.element_id] = vm
    return values


def extract_point_displacements(
    model: Model,
    result: StaticSolveResult,
    points: list[PointOfInterest],
) -> list[GeotechPointResult]:
    """Extract nearest-node displacement for named points of interest."""
    if not points:
        return []

    node_displacement_by_id = {item.node_id: item for item in result.node_displacements}
    extracted: list[GeotechPointResult] = []
    for point in points:
        best_node = None
        best_distance = float("inf")
        for node in model.mesh.nodes:
            dist = float(np.hypot(node.x - point.x, node.y - point.y))
            if dist < best_distance:
                best_distance = dist
                best_node = node
        if best_node is None:
            raise RuntimeError("Point extraction failed because model has no nodes.")
        disp = node_displacement_by_id[best_node.id]
        extracted.append(
            GeotechPointResult(
                name=point.name,
                target_x=point.x,
                target_y=point.y,
                node_id=best_node.id,
                node_x=best_node.x,
                node_y=best_node.y,
                ux=disp.ux,
                uy=disp.uy,
                distance=best_distance,
            )
        )
    return extracted


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _build_svg_transform(
    model: Model,
    *,
    width: float,
    height: float,
    padding: float,
) -> tuple[float, float, float]:
    xs = [node.x for node in model.mesh.nodes]
    ys = [node.y for node in model.mesh.nodes]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)
    scale = min((width - 2.0 * padding) / span_x, (height - 2.0 * padding) / span_y)
    return min_x, min_y, scale


def _map_point(
    x: float,
    y: float,
    *,
    min_x: float,
    min_y: float,
    scale: float,
    height: float,
    padding: float,
) -> tuple[float, float]:
    sx = padding + (x - min_x) * scale
    sy = height - padding - (y - min_y) * scale
    return sx, sy


def write_element_cloud_svg(
    model: Model,
    scalar_by_element_id: dict[int, float],
    output_path: str | Path,
    *,
    title: str,
    points_of_interest: list[GeotechPointResult] | None = None,
) -> Path:
    """Write a lightweight SVG cloud plot from element scalar values."""
    if not scalar_by_element_id:
        raise ValueError("scalar_by_element_id cannot be empty.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    width = 1080.0
    height = 640.0
    padding = 54.0
    min_x, min_y, scale = _build_svg_transform(model, width=width, height=height, padding=padding)

    values = list(scalar_by_element_id.values())
    vmin = float(min(values))
    vmax = float(max(values))
    if abs(vmax - vmin) <= 1e-14:
        pad = max(abs(vmax), 1.0) * 0.01
        vmin -= pad
        vmax += pad

    node_by_id = {node.id: node for node in model.mesh.nodes}
    polygon_lines: list[str] = []
    for element in model.mesh.elements:
        value = scalar_by_element_id.get(element.id)
        if value is None:
            continue
        rgb = _color_for_value(value, vmin, vmax)
        points: list[str] = []
        for node_id in element.connectivity:
            node = node_by_id[node_id]
            sx, sy = _map_point(
                node.x,
                node.y,
                min_x=min_x,
                min_y=min_y,
                scale=scale,
                height=height,
                padding=padding,
            )
            points.append(f"{sx:.3f},{sy:.3f}")
        polygon_lines.append(
            f'<polygon points="{" ".join(points)}" '
            f'style="fill:rgb({rgb[0]},{rgb[1]},{rgb[2]});stroke:rgb(23,37,84);stroke-width:0.8" />'
        )

    poi_lines: list[str] = []
    for poi in points_of_interest or []:
        sx, sy = _map_point(
            poi.node_x,
            poi.node_y,
            min_x=min_x,
            min_y=min_y,
            scale=scale,
            height=height,
            padding=padding,
        )
        poi_lines.append(
            f'<circle cx="{sx:.3f}" cy="{sy:.3f}" r="4.5" style="fill:#111827;stroke:#f59e0b;stroke-width:1.5" />'
        )
        poi_lines.append(
            f'<text x="{sx + 8.0:.3f}" y="{sy - 8.0:.3f}" '
            f'style="font:13px Segoe UI; fill:#111827">{poi.name}</text>'
        )

    svg_text = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}">',
            '<rect x="0" y="0" width="100%" height="100%" fill="#eef2f7" />',
            f'<text x="20" y="30" style="font:600 18px Segoe UI; fill:#0f172a">{title}</text>',
            f'<text x="20" y="54" style="font:13px Segoe UI; fill:#334155">min={vmin:.6e}, max={vmax:.6e}</text>',
            *polygon_lines,
            *poi_lines,
            "</svg>",
            "",
        ]
    )
    out.write_text(svg_text, encoding="utf-8")
    return out


def export_geotech_result_artifacts(
    model: Model,
    result: StaticSolveResult,
    point_results: list[GeotechPointResult],
    von_mises_by_element_id: dict[int, float],
    output_dir: str | Path,
) -> GeotechOutputFiles:
    """Export workflow output to files for class reports and UI integration."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    node_csv = base / "node_displacements.csv"
    elem_csv = base / "element_results.csv"
    point_csv = base / "point_displacements.csv"
    summary_json = base / "summary.json"
    ux_svg = base / "ux_cloud.svg"
    uy_svg = base / "uy_cloud.svg"
    vm_svg = base / "von_mises_cloud.svg"

    _write_csv(
        node_csv,
        ["node_id", "ux", "uy"],
        [[item.node_id, item.ux, item.uy] for item in result.node_displacements],
    )
    _write_csv(
        elem_csv,
        ["element_id", "ex", "ey", "gxy", "sx", "sy", "txy", "von_mises"],
        [
            [
                item.element_id,
                item.strain[0],
                item.strain[1],
                item.strain[2],
                item.stress[0],
                item.stress[1],
                item.stress[2],
                von_mises_by_element_id[item.element_id],
            ]
            for item in result.element_results
        ],
    )
    _write_csv(
        point_csv,
        ["name", "target_x", "target_y", "node_id", "node_x", "node_y", "ux", "uy", "distance"],
        [
            [
                item.name,
                item.target_x,
                item.target_y,
                item.node_id,
                item.node_x,
                item.node_y,
                item.ux,
                item.uy,
                item.distance,
            ]
            for item in point_results
        ],
    )

    node_disp_by_id = {item.node_id: item for item in result.node_displacements}
    element_by_id = {element.id: element for element in model.mesh.elements}
    ux_by_element: dict[int, float] = {}
    uy_by_element: dict[int, float] = {}
    for item in result.element_results:
        element = element_by_id.get(item.element_id)
        if element is None:
            raise RuntimeError(
                f"Element result {item.element_id} does not exist in model mesh."
            )
        ux_values = [node_disp_by_id[node_id].ux for node_id in element.connectivity]
        uy_values = [node_disp_by_id[node_id].uy for node_id in element.connectivity]
        ux_by_element[item.element_id] = float(np.mean(ux_values))
        uy_by_element[item.element_id] = float(np.mean(uy_values))

    write_element_cloud_svg(model, ux_by_element, ux_svg, title="Ux Cloud Plot", points_of_interest=point_results)
    write_element_cloud_svg(model, uy_by_element, uy_svg, title="Uy Cloud Plot", points_of_interest=point_results)
    write_element_cloud_svg(model, von_mises_by_element_id, vm_svg, title="Von Mises Cloud Plot", points_of_interest=point_results)

    summary_payload = {
        "summary": asdict(result.summary),
        "point_results": [asdict(item) for item in point_results],
        "output_files": {
            "node_displacements_csv": str(node_csv),
            "element_results_csv": str(elem_csv),
            "point_displacements_csv": str(point_csv),
            "ux_cloud_svg": str(ux_svg),
            "uy_cloud_svg": str(uy_svg),
            "von_mises_cloud_svg": str(vm_svg),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return GeotechOutputFiles(
        base_dir=base,
        summary_json=summary_json,
        node_displacements_csv=node_csv,
        element_results_csv=elem_csv,
        point_displacements_csv=point_csv,
        ux_cloud_svg=ux_svg,
        uy_cloud_svg=uy_svg,
        von_mises_cloud_svg=vm_svg,
    )
