from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from fem_ai_solver.fem.services import solve_linear_static
from fem_ai_solver.mesh.importers import (
    load_t3_mesh_from_gmsh_msh,
    load_t3_mesh_from_gmsh_msh_with_physical_groups,
    load_t3_mesh_from_json,
)
from fem_ai_solver.preprocessing.demo_model import (
    apply_left_clamp_right_nodal_force,
    model_from_t3_mesh,
)
from fem_ai_solver.preprocessing.model_services import apply_default_templates_from_physical_groups

_RUNTIME_DIR = Path("tests/.tmp_runtime")
_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _runtime_file(prefix: str, suffix: str, content: str):
    path = _RUNTIME_DIR / f"{prefix}_{uuid4().hex}{suffix}"
    path.write_text(content, encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_load_t3_mesh_from_json() -> None:
    payload = {
        "nodes": [
            {"id": 1, "x": 0.0, "y": 0.0},
            {"id": 2, "x": 1.0, "y": 0.0},
            {"id": 3, "x": 1.0, "y": 1.0},
            {"id": 4, "x": 0.0, "y": 1.0},
        ],
        "elements": [
            {"id": 1, "connectivity": [1, 2, 3], "material_id": 1},
            {"id": 2, "connectivity": [1, 3, 4], "material_id": 1},
        ],
    }

    with _runtime_file("mesh", ".json", json.dumps(payload)) as path:
        mesh = load_t3_mesh_from_json(path)

    assert len(mesh.nodes) == 4
    assert len(mesh.elements) == 2
    assert mesh.elements[0].type == "T3"


def test_load_t3_mesh_from_gmsh_msh_stage1_compatibility() -> None:
    gmsh_text = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
$Elements
2
1 2 0 1 2 3
2 2 0 1 3 4
$EndElements
"""

    with _runtime_file("mesh", ".msh", gmsh_text) as path:
        mesh = load_t3_mesh_from_gmsh_msh(path, material_id=9)

    assert len(mesh.nodes) == 4
    assert len(mesh.elements) == 2
    assert mesh.elements[0].material_id == 9
    assert mesh.elements[1].connectivity == [1, 3, 4]


def test_gmsh_physical_group_mapping_to_material_bc_and_load_templates() -> None:
    gmsh_text = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 11 "MAT_STEEL"
1 21 "BC_FIX"
1 22 "LOAD_FX=300"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
$Elements
4
1 1 2 21 1 1 4
2 1 2 22 2 2 3
3 2 2 11 3 1 2 3
4 2 2 11 4 1 3 4
$EndElements
"""

    with _runtime_file("mesh", ".msh", gmsh_text) as path:
        gmsh_import = load_t3_mesh_from_gmsh_msh_with_physical_groups(path)

    assert gmsh_import.physical_name_by_dim_tag[(2, 11)] == "MAT_STEEL"
    assert gmsh_import.line_node_ids_by_physical_tag[21] == [1, 4]
    assert gmsh_import.line_node_ids_by_physical_tag[22] == [2, 3]

    model = model_from_t3_mesh(gmsh_import.mesh, preserve_element_material_ids=True)
    report = apply_default_templates_from_physical_groups(model, gmsh_import)

    assert report.material_count == 1
    assert report.boundary_condition_count == 4
    assert report.load_count == 2

    assert {material.id for material in model.materials} == {11}

    bc_map = {(bc.node_id, bc.dof): bc.value for bc in model.boundary_conditions}
    assert bc_map[(1, "ux")] == 0.0
    assert bc_map[(1, "uy")] == 0.0
    assert bc_map[(4, "ux")] == 0.0
    assert bc_map[(4, "uy")] == 0.0

    load_map = {(load.node_id, load.dof): load.value for load in model.loads}
    assert load_map[(2, "fx")] == 150.0
    assert load_map[(3, "fx")] == 150.0

    result = solve_linear_static(model, backend="python")
    assert result.summary.element_count == 2
    assert result.summary.node_count == 4


def test_imported_mesh_model_can_solve_python_backend() -> None:
    gmsh_text = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
$Elements
2
1 2 0 1 2 3
2 2 0 1 3 4
$EndElements
"""

    with _runtime_file("mesh", ".msh", gmsh_text) as path:
        mesh = load_t3_mesh_from_gmsh_msh(path)

    model = model_from_t3_mesh(mesh)
    apply_left_clamp_right_nodal_force(model, fx_total=500.0, fy_top=-100.0)

    result = solve_linear_static(model, backend="python")

    assert result.summary.node_count == 4
    assert result.summary.element_count == 2
    assert result.displacements.shape[0] == 8
