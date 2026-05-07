from __future__ import annotations

from pathlib import Path

from fem_ai_solver.fem.services import solve_linear_static
from fem_ai_solver.preprocessing.model_services import validate_model

from .meshing import build_layered_template_model
from .models import GeotechTemplateInput, GeotechTemplateResult
from .postprocessing import (
    compute_von_mises_by_element,
    export_geotech_result_artifacts,
    extract_point_displacements,
)


def run_geotech_template_workflow(
    template: GeotechTemplateInput,
    *,
    output_dir: str | Path | None = None,
) -> GeotechTemplateResult:
    """Execute end-to-end 2D geotechnical template workflow.

    Pipeline:
    1) Layered boundary input -> automatic T3 mesh and model generation.
    2) Apply template BCs and top line load.
    3) Call existing dual-backend solver (`python` or `cpp`).
    4) Recover A/B (or custom) point displacements and Von Mises.
    5) Optionally export CSV/SVG artifacts.
    """
    build = build_layered_template_model(template)
    model = build.model

    # Reuse the existing project-side validator to keep exception semantics aligned.
    validate_model(model, area_tolerance=template.area_tolerance)

    result = solve_linear_static(
        model,
        thickness=template.thickness,
        area_tolerance=template.area_tolerance,
        backend=template.backend,
    )

    point_results = extract_point_displacements(model, result, template.points_of_interest)
    von_mises = compute_von_mises_by_element(result)

    files = None
    if output_dir is not None:
        files = export_geotech_result_artifacts(
            model,
            result,
            point_results,
            von_mises,
            output_dir,
        )

    return GeotechTemplateResult(
        model=model,
        solve_result=result,
        point_results=point_results,
        von_mises_by_element_id=von_mises,
        output_files=files,
    )
