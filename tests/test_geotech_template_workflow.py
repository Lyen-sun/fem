from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fem_ai_solver.ai.geotech_assistant import (
    detect_geotech_from_image,
    generate_geotech_execution_script_from_text,
)
from fem_ai_solver.geotech import (
    GeotechTemplateInput,
    LayerBoundary,
    PointOfInterest,
    build_template_input_from_text_description,
    load_geotech_layers_from_csv,
    run_geotech_template_workflow,
)
from fem_ai_solver.geotech.models import GeotechLayer
from fem_ai_solver.fem.model import Material


def _sample_text_description() -> str:
    return """
layer Fill: top=[(0,0),(20,0)] bottom=[(0,-3),(20,-4)] E=2.0e7 nu=0.30
layer Clay: top=[(0,-3),(20,-4)] bottom=[(0,-10),(20,-10)] E=3.7e7 nu=0.29
Q=30
mesh_size=1.0
point A=(2,-2.5)
point B=(14,-3.2)
"""


def _runtime_dir(name: str) -> Path:
    base = Path("tests/.tmp_runtime/geotech_workflow") / name
    base.mkdir(parents=True, exist_ok=True)
    return base


def test_geotech_template_text_pipeline_python_backend() -> None:
    out_dir = _runtime_dir("text_pipeline")
    template = build_template_input_from_text_description(_sample_text_description(), backend="python")
    result = run_geotech_template_workflow(template, output_dir=out_dir)

    assert result.solve_result.summary.node_count > 0
    assert result.solve_result.summary.element_count > 0
    assert np.all(np.isfinite(result.solve_result.displacements))
    assert np.all(np.isfinite(result.solve_result.reactions))
    assert len(result.point_results) == 2
    assert {item.name for item in result.point_results} == {"A", "B"}
    assert result.output_files is not None

    assert result.output_files.node_displacements_csv.exists()
    assert result.output_files.element_results_csv.exists()
    assert result.output_files.point_displacements_csv.exists()
    assert result.output_files.ux_cloud_svg.exists()
    assert result.output_files.uy_cloud_svg.exists()
    assert result.output_files.von_mises_cloud_svg.exists()
    assert result.output_files.summary_json.exists()

    assert "node_id,ux,uy" in result.output_files.node_displacements_csv.read_text(encoding="utf-8")
    assert "element_id,ex,ey,gxy,sx,sy,txy,von_mises" in result.output_files.element_results_csv.read_text(
        encoding="utf-8"
    )


def test_load_geotech_layers_from_csv_and_run() -> None:
    work_dir = _runtime_dir("csv_pipeline")
    csv_path = work_dir / "layers.csv"
    csv_path.write_text(
        "\n".join(
            [
                "layer,boundary,point_order,x,y,material_id,young_modulus,poisson_ratio,plane_stress",
                "Fill,top,1,0,0,1,2.0e7,0.30,false",
                "Fill,top,2,16,0,1,2.0e7,0.30,false",
                "Fill,bottom,1,0,-2,1,2.0e7,0.30,false",
                "Fill,bottom,2,16,-3,1,2.0e7,0.30,false",
                "Rock,top,1,0,-2,2,5.0e7,0.27,false",
                "Rock,top,2,16,-3,2,5.0e7,0.27,false",
                "Rock,bottom,1,0,-12,2,5.0e7,0.27,false",
                "Rock,bottom,2,16,-12,2,5.0e7,0.27,false",
            ]
        ),
        encoding="utf-8",
    )

    layers = load_geotech_layers_from_csv(csv_path)
    template = GeotechTemplateInput(
        layers=layers,
        mesh_target_size=0.8,
        top_line_load=18.0,
        points_of_interest=[PointOfInterest(name="A", x=1.0, y=-1.0)],
        backend="python",
    )
    result = run_geotech_template_workflow(template)
    assert result.solve_result.summary.element_count > 10
    assert len(result.point_results) == 1


def test_geotech_template_rejects_overlapping_layers() -> None:
    layers = [
        GeotechLayer(
            name="Upper",
            material=Material(id=1, young_modulus=2.0e7, poisson_ratio=0.30, plane_stress=False),
            top_boundary=LayerBoundary(points=[(0.0, 0.0), (10.0, 0.0)]),
            bottom_boundary=LayerBoundary(points=[(0.0, -4.0), (10.0, -4.0)]),
        ),
        GeotechLayer(
            name="Lower",
            material=Material(id=2, young_modulus=4.0e7, poisson_ratio=0.28, plane_stress=False),
            top_boundary=LayerBoundary(points=[(0.0, -3.0), (10.0, -3.0)]),  # overlaps with Upper
            bottom_boundary=LayerBoundary(points=[(0.0, -10.0), (10.0, -10.0)]),
        ),
    ]
    template = GeotechTemplateInput(layers=layers, mesh_target_size=1.0, top_line_load=10.0, backend="python")
    with pytest.raises(ValueError, match="overlap"):
        run_geotech_template_workflow(template)


def test_optional_image_detection_placeholder() -> None:
    image_path = _runtime_dir("image_placeholder") / "placeholder.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(NotImplementedError, match="Image recognition"):
        detect_geotech_from_image(image_path)


def test_generate_execution_script_from_text() -> None:
    script = generate_geotech_execution_script_from_text(
        _sample_text_description(),
        backend="cpp",
        output_dir="output/case_a",
    )
    assert "build_template_input_from_text_description" in script
    assert "backend='cpp'" in script
    assert "output/case_a" in script


def test_geotech_template_cpp_backend_parity_if_available() -> None:
    pytest.importorskip("fem_core", reason="fem_core extension is not built in this environment")
    text = _sample_text_description()
    py_result = run_geotech_template_workflow(
        build_template_input_from_text_description(text, backend="python")
    )
    cpp_result = run_geotech_template_workflow(
        build_template_input_from_text_description(text, backend="cpp")
    )

    np.testing.assert_allclose(
        cpp_result.solve_result.displacements,
        py_result.solve_result.displacements,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        cpp_result.solve_result.reactions,
        py_result.solve_result.reactions,
        rtol=1e-8,
        atol=1e-5,
    )
    assert cpp_result.von_mises_by_element_id.keys() == py_result.von_mises_by_element_id.keys()
