from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from fem_ai_solver.mesh.importers import load_t3_mesh_from_gmsh_msh_with_physical_groups
from fem_ai_solver.preprocessing.demo_model import build_stage1_demo_model, model_from_t3_mesh
from fem_ai_solver.preprocessing.model_services import (
    apply_default_templates_from_physical_groups,
    build_physical_group_mapping_preview,
    collect_pre_solve_warnings,
    summarize_mapping_entries,
    summarize_model_state,
    validate_model,
)

_RUNTIME_DIR = Path("tests/.tmp_runtime")
_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _runtime_msh(content: str):
    path = _RUNTIME_DIR / f"mapping_{uuid4().hex}.msh"
    path.write_text(content, encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _sample_gmsh_with_mixed_groups() -> str:
    return """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
4
2 11 "MAT_CORE"
1 21 "BC_FIX"
1 22 "LOAD_FX=300"
1 23 "UNKNOWN_RULE"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
$Elements
5
1 1 2 21 1 1 4
2 1 2 22 2 2 3
3 1 2 23 3 1 2
4 2 2 11 4 1 2 3
5 2 2 11 5 1 3 4
$EndElements
"""


def test_summarize_model_state_reports_counts_and_backend() -> None:
    model = build_stage1_demo_model()

    summary = summarize_model_state(model, backend="cpp")

    assert summary["node_count"] == len(model.mesh.nodes)
    assert summary["element_count"] == len(model.mesh.elements)
    assert summary["material_count"] == len(model.materials)
    assert summary["backend"] == "cpp"
    assert summary["constrained_node_count"] > 0
    assert summary["loaded_node_count"] > 0


def test_validate_model_accepts_stage1_demo_model() -> None:
    model = build_stage1_demo_model()
    validate_model(model)


def test_validate_model_rejects_unknown_load_node() -> None:
    model = build_stage1_demo_model()
    model.loads[0].node_id = 9999

    with pytest.raises(ValueError, match="Load references unknown node id"):
        validate_model(model)


def test_mapping_preview_and_summary_reports_group_statuses() -> None:
    with _runtime_msh(_sample_gmsh_with_mixed_groups()) as path:
        gmsh_import = load_t3_mesh_from_gmsh_msh_with_physical_groups(path)

    entries = build_physical_group_mapping_preview(gmsh_import)
    summary = summarize_mapping_entries(entries)

    assert summary["total_group_count"] >= 4
    assert summary["mapped_group_count"] >= 2
    assert summary["unrecognized_group_count"] >= 1

    by_name = {entry.name: entry for entry in entries}
    assert by_name["MAT_CORE"].mapped_to == "material_id=11"
    assert by_name["BC_FIX"].status == "mapped"
    assert by_name["LOAD_FX=300"].mapped_to.startswith("Load fx")
    assert by_name["UNKNOWN_RULE"].status == "unrecognized"


def test_template_application_report_contains_mapping_entries() -> None:
    with _runtime_msh(_sample_gmsh_with_mixed_groups()) as path:
        gmsh_import = load_t3_mesh_from_gmsh_msh_with_physical_groups(path)

    model = model_from_t3_mesh(gmsh_import.mesh, preserve_element_material_ids=True)
    report = apply_default_templates_from_physical_groups(model, gmsh_import)

    assert report.material_count == 1
    assert report.boundary_condition_count == 4
    assert report.load_count == 2
    assert report.mapping_entries
    assert report.recognized_group_count >= 3


def test_collect_pre_solve_warnings_flags_empty_bc_and_loads() -> None:
    model = build_stage1_demo_model()
    model.boundary_conditions = []
    model.loads = []

    warnings = collect_pre_solve_warnings(model)

    assert any("boundary condition" in message for message in warnings)
    assert any("nodal loads" in message for message in warnings)
