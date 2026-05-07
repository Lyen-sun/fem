from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(slots=True)
class ParsedLayerIntent:
    """Layer intent parsed from free text."""

    name: str
    top_points: list[tuple[float, float]]
    bottom_points: list[tuple[float, float]]
    young_modulus: float
    poisson_ratio: float


@dataclass(slots=True)
class ParsedPointIntent:
    """Point-of-interest intent parsed from text."""

    name: str
    x: float
    y: float


@dataclass(slots=True)
class ParsedGeotechIntent:
    """Structured output from text parsing for template workflow automation."""

    layers: list[ParsedLayerIntent] = field(default_factory=list)
    top_line_load: float = 0.0
    mesh_target_size: float = 1.0
    points_of_interest: list[ParsedPointIntent] = field(default_factory=list)


def _parse_point_list(text: str) -> list[tuple[float, float]]:
    """Parse '(x,y);(x,y)' or '[x,y; x,y]' style point lists."""
    point_pairs = re.findall(
        rf"\(\s*({_FLOAT_PATTERN})\s*,\s*({_FLOAT_PATTERN})\s*\)",
        text,
    )
    if not point_pairs:
        point_pairs = re.findall(
            rf"({_FLOAT_PATTERN})\s*,\s*({_FLOAT_PATTERN})",
            text,
        )
    if not point_pairs:
        raise ValueError(f"Failed to parse points from: {text!r}")
    return [(float(x), float(y)) for x, y in point_pairs]


def parse_geotech_text_description(text: str) -> ParsedGeotechIntent:
    """Parse a lightweight geotechnical description into structured intents.

    Supported line examples:
    - layer Fill: top=[(0,0),(40,0)] bottom=[(0,-6),(40,-8)] E=20e6 nu=0.30
    - point A=(2,-1.5)
    - point B=(25,-2.0)
    - Q=35
    - mesh_size=0.8
    """
    intent = ParsedGeotechIntent()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        layer_match = re.match(
            rf"(?i)layer\s+([^:]+)\s*:\s*top\s*=\s*(.+?)\s+bottom\s*=\s*(.+?)\s+E\s*=\s*({_FLOAT_PATTERN})\s+nu\s*=\s*({_FLOAT_PATTERN})\s*$",
            line,
        )
        if layer_match:
            name = layer_match.group(1).strip()
            top_points = _parse_point_list(layer_match.group(2))
            bottom_points = _parse_point_list(layer_match.group(3))
            young_modulus = float(layer_match.group(4))
            poisson_ratio = float(layer_match.group(5))
            intent.layers.append(
                ParsedLayerIntent(
                    name=name,
                    top_points=top_points,
                    bottom_points=bottom_points,
                    young_modulus=young_modulus,
                    poisson_ratio=poisson_ratio,
                )
            )
            continue

        point_match = re.match(
            rf"(?i)point\s+([A-Za-z0-9_]+)\s*=\s*\(\s*({_FLOAT_PATTERN})\s*,\s*({_FLOAT_PATTERN})\s*\)\s*$",
            line,
        )
        if point_match:
            intent.points_of_interest.append(
                ParsedPointIntent(
                    name=point_match.group(1),
                    x=float(point_match.group(2)),
                    y=float(point_match.group(3)),
                )
            )
            continue

        q_match = re.match(rf"(?i)(?:q|top_line_load)\s*=\s*({_FLOAT_PATTERN})\s*$", line)
        if q_match:
            intent.top_line_load = float(q_match.group(1))
            continue

        mesh_match = re.match(rf"(?i)(?:mesh_size|mesh_target_size)\s*=\s*({_FLOAT_PATTERN})\s*$", line)
        if mesh_match:
            intent.mesh_target_size = float(mesh_match.group(1))
            continue

    if not intent.layers:
        raise ValueError(
            "No layer definitions were parsed from text. "
            "Expected lines like: "
            "'layer Fill: top=[(0,0),(40,0)] bottom=[(0,-6),(40,-8)] E=20e6 nu=0.30'."
        )
    return intent


def detect_geotech_from_image(image_path: str | Path) -> ParsedGeotechIntent:
    """Optional image-recognition entry point.

    This project stage keeps image parsing as an extension point. A production
    implementation can use OpenCV + segmentation models to detect interfaces and
    point labels, then return `ParsedGeotechIntent`.
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"image file does not exist: {path}")
    raise NotImplementedError(
        "Image recognition is optional and not enabled in this lightweight baseline. "
        "Use CSV/text inputs, or plug in a detector that returns ParsedGeotechIntent."
    )


def generate_geotech_execution_script_from_text(
    text: str,
    *,
    backend: str = "python",
    output_dir: str = "output/geotech_auto",
) -> str:
    """Generate a runnable script that executes the template workflow.

    This helper is intended for AI-assisted automation:
    LLM/vision module -> structured text -> script generation -> execution.
    """
    # Validate the text once so script generation fails early if text is invalid.
    parse_geotech_text_description(text)
    escaped_text = text.strip().replace('"""', '\\"\\"\\"')

    return f"""from fem_ai_solver.geotech import build_template_input_from_text_description, run_geotech_template_workflow

description = \"\"\"{escaped_text}\"\"\"

template = build_template_input_from_text_description(description, backend={backend!r})
result = run_geotech_template_workflow(template, output_dir={output_dir!r})

print("Solved:", result.solve_result.summary)
for point in result.point_results:
    print(point.name, point.ux, point.uy)
"""
