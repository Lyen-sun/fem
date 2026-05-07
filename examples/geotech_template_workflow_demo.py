from __future__ import annotations

import argparse
from pathlib import Path

from fem_ai_solver.geotech import (
    PointOfInterest,
    build_template_input_from_text_description,
    run_geotech_template_workflow,
)


def _default_description() -> str:
    # Text format intentionally stays lightweight, so course users can edit it
    # directly without needing a full CAD preprocessor.
    return """
# layer format:
# layer <name>: top=[(x,y),(x,y)] bottom=[(x,y),(x,y)] E=<YoungModulus> nu=<Poisson>
layer Fill: top=[(0,0),(30,0)] bottom=[(0,-4),(30,-6)] E=2.0e7 nu=0.30
layer ClayRock: top=[(0,-4),(30,-6)] bottom=[(0,-12),(30,-12)] E=3.7e7 nu=0.29
layer WeatheredRock: top=[(0,-12),(30,-12)] bottom=[(0,-20),(30,-20)] E=5.0e7 nu=0.27
Q=35
mesh_size=1.2
point A=(2.0,-3.8)
point B=(20.0,-5.5)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 2D geotechnical template FEM workflow demo.")
    parser.add_argument(
        "--backend",
        default="python",
        choices=["python", "cpp"],
        help="Solver backend.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/output/geotech_demo",
        help="Directory for CSV and cloud plot outputs.",
    )
    args = parser.parse_args()

    template = build_template_input_from_text_description(
        _default_description(),
        backend=args.backend,
    )

    # Keep default A/B points from text; only add fallback entries if missing.
    existing_names = {item.name for item in template.points_of_interest}
    if "A" not in existing_names:
        template.points_of_interest.append(PointOfInterest(name="A", x=2.0, y=-3.8))
    if "B" not in existing_names:
        template.points_of_interest.append(PointOfInterest(name="B", x=20.0, y=-5.5))

    run_result = run_geotech_template_workflow(template, output_dir=args.output_dir)
    summary = run_result.solve_result.summary

    print(f"[geotech-demo] backend={args.backend}")
    print(
        f"[geotech-demo] nodes={summary.node_count}, elements={summary.element_count}, "
        f"max|u|={summary.max_displacement:.6e}"
    )
    for point in run_result.point_results:
        print(
            f"[geotech-demo] {point.name}: ux={point.ux:.6e}, uy={point.uy:.6e}, "
            f"nearest_node={point.node_id}, distance={point.distance:.6e}"
        )

    if run_result.output_files is not None:
        print(f"[geotech-demo] outputs -> {Path(run_result.output_files.base_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
