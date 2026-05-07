from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtWidgets import QApplication

from fem_ai_solver.fem.model import Element, Mesh, Node
from fem_ai_solver.mesh.workflow import (
    BuiltinT3Mesher,
    GmshMesher,
    MeshGeometryEdge,
    MeshGeometryPoint,
    MeshGenerationRequest,
    MeshRegion,
    MeshSeed,
    delete_elements_from_mesh,
    evaluate_t3_mesh_quality,
)
from fem_ai_solver.preprocessing.scene_model import GeometrySetDef, LoadDefinition
from fem_ai_solver.ui.main_window import MainWindow


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_builtin_mesher_region_seed_increases_mesh_density() -> None:
    mesher = BuiltinT3Mesher()
    region = MeshRegion(
        id="region:a",
        name="A",
        points=[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
        material_id=1,
    )

    coarse = mesher.generate(
        MeshGenerationRequest(
            regions=[region],
            seed=MeshSeed(global_size=0.8),
            operation="generate",
        )
    )
    fine = mesher.generate(
        MeshGenerationRequest(
            regions=[region],
            seed=MeshSeed(global_size=0.8, region_sizes={"region:a": 0.25}),
            operation="generate",
        )
    )

    assert len(fine.mesh.nodes) > len(coarse.mesh.nodes)
    assert len(fine.mesh.elements) > len(coarse.mesh.elements)


def test_builtin_mesher_edge_seed_adds_boundary_nodes() -> None:
    mesher = BuiltinT3Mesher()
    region = MeshRegion(
        id="region:a",
        name="A",
        points=[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
        edge_ids=["edge:bottom", "edge:right", "edge:top", "edge:left"],
        material_id=1,
    )
    result = mesher.generate(
        MeshGenerationRequest(
            regions=[region],
            seed=MeshSeed(global_size=1.0, edge_seeds={"edge:bottom": 0.25}),
            operation="generate",
        )
    )

    assert "edge:bottom" in result.edge_to_node_ids
    assert len(result.edge_to_node_ids["edge:bottom"]) >= 5


def test_builtin_mesher_returns_real_geometry_point_mapping() -> None:
    mesher = BuiltinT3Mesher()
    region = MeshRegion(
        id="region:a",
        name="A",
        points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        point_ids=["p0", "p1", "p2", "p3"],
        edge_ids=["e0", "e1", "e2", "e3"],
        material_id=1,
    )

    result = mesher.generate(
        MeshGenerationRequest(
            regions=[region],
            seed=MeshSeed(global_size=0.5),
            geometry_points={
                "p0": MeshGeometryPoint(id="p0", x=0.0, y=0.0),
                "p1": MeshGeometryPoint(id="p1", x=1.0, y=0.0),
                "p2": MeshGeometryPoint(id="p2", x=1.0, y=1.0),
                "p3": MeshGeometryPoint(id="p3", x=0.0, y=1.0),
            },
            operation="generate",
        )
    )

    assert set(result.point_to_node_ids) >= {"p0", "p1", "p2", "p3"}
    assert all(len(result.point_to_node_ids[point_id]) == 1 for point_id in ("p0", "p1", "p2", "p3"))


def test_gmsh_mesher_preserves_geometry_mapping_when_available() -> None:
    pytest.importorskip("gmsh")
    mesher = GmshMesher()
    region = MeshRegion(
        id="region:a",
        name="A",
        points=[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
        point_ids=["p0", "p1", "p2", "p3"],
        edge_ids=["e0", "e1", "e2", "e3"],
        material_id=1,
    )

    result = mesher.generate(
        MeshGenerationRequest(
            regions=[region],
            seed=MeshSeed(global_size=0.35, edge_seeds={"e0": 0.25}),
            geometry_points={
                "p0": MeshGeometryPoint(id="p0", x=0.0, y=0.0),
                "p1": MeshGeometryPoint(id="p1", x=2.0, y=0.0),
                "p2": MeshGeometryPoint(id="p2", x=2.0, y=1.0),
                "p3": MeshGeometryPoint(id="p3", x=0.0, y=1.0),
            },
            geometry_edges={
                "e0": MeshGeometryEdge(id="e0", start_point_id="p0", end_point_id="p1"),
                "e1": MeshGeometryEdge(id="e1", start_point_id="p1", end_point_id="p2"),
                "e2": MeshGeometryEdge(id="e2", start_point_id="p2", end_point_id="p3"),
                "e3": MeshGeometryEdge(id="e3", start_point_id="p3", end_point_id="p0"),
            },
            operation="generate",
        )
    )

    assert len(result.mesh.elements) > 0
    assert result.region_to_element_ids["region:a"]
    assert len(result.point_to_node_ids["p0"]) == 1
    assert len(result.edge_to_node_ids["e0"]) >= 3


def test_gmsh_remesh_regenerates_full_geometry_when_available() -> None:
    pytest.importorskip("gmsh")
    mesher = GmshMesher()
    left = MeshRegion(
        id="region:left",
        name="Left",
        points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        point_ids=["lp0", "lp1", "lp2", "lp3"],
        edge_ids=["le0", "le1", "le2", "le3"],
        material_id=1,
    )
    right = MeshRegion(
        id="region:right",
        name="Right",
        points=[(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)],
        point_ids=["rp0", "rp1", "rp2", "rp3"],
        edge_ids=["re0", "re1", "re2", "re3"],
        material_id=2,
    )
    geometry_points = {
        point_id: MeshGeometryPoint(id=point_id, x=x, y=y)
        for point_id, (x, y) in {
            "lp0": (0.0, 0.0),
            "lp1": (1.0, 0.0),
            "lp2": (1.0, 1.0),
            "lp3": (0.0, 1.0),
            "rp0": (2.0, 0.0),
            "rp1": (3.0, 0.0),
            "rp2": (3.0, 1.0),
            "rp3": (2.0, 1.0),
        }.items()
    }
    initial = mesher.generate(
        MeshGenerationRequest(
            regions=[left, right],
            seed=MeshSeed(global_size=0.5),
            geometry_points=geometry_points,
            operation="generate",
        )
    )

    remeshed = mesher.generate(
        MeshGenerationRequest(
            regions=[left, right],
            seed=MeshSeed(global_size=0.4),
            geometry_points=geometry_points,
            operation="remesh",
            existing_mesh=initial.mesh,
            target_region_ids={"region:left"},
        )
    )

    assert set(remeshed.region_to_element_ids) == {"region:left", "region:right"}
    assert any("full geometry" in warning for warning in remeshed.warnings)


def test_delete_elements_from_mesh_cleans_orphans_but_preserves_requested_nodes() -> None:
    mesh = Mesh(
        nodes=[
            Node(id=1, x=0.0, y=0.0),
            Node(id=2, x=1.0, y=0.0),
            Node(id=3, x=0.0, y=1.0),
            Node(id=4, x=2.0, y=2.0),
        ],
        elements=[Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=1)],
    )

    reduced = delete_elements_from_mesh(mesh, {1}, preserve_node_ids={4})

    assert reduced.elements == []
    assert [node.id for node in reduced.nodes] == [4]


def test_quality_report_flags_bad_triangles() -> None:
    mesh = Mesh(
        nodes=[
            Node(id=1, x=0.0, y=0.0),
            Node(id=2, x=1.0, y=0.0),
            Node(id=3, x=0.0, y=1.0),
            Node(id=4, x=10.0, y=0.0),
            Node(id=5, x=12.0, y=0.0),
            Node(id=6, x=10.0001, y=1.0e-5),
        ],
        elements=[
            Element(id=1, type="T3", connectivity=[1, 3, 2], material_id=1),
            Element(id=2, type="T3", connectivity=[4, 5, 6], material_id=1),
        ],
    )

    report = evaluate_t3_mesh_quality(mesh)

    assert report.bad_element_count >= 2
    assert 1 in report.bad_element_ids
    assert 2 in report.bad_element_ids


def test_mainwindow_mesh_region_seed_and_delete_flow() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()
        window._selected_face_region_sketch_keys = {"active"}
        window._apply_mesh_size_to_selected_regions()
        window._generate_selected_region_mesh()

        assert window._model is not None
        assert len(window._model.mesh.elements) > 0
        assert window._last_mesh_quality_report is not None
        generated_count = len(window._model.mesh.elements)

        first_element_id = int(window._model.mesh.elements[0].id)
        window._selected_element_ids = {first_element_id}
        window._delete_selected_mesh_elements()

        assert window._model is not None
        assert len(window._model.mesh.elements) < generated_count
    finally:
        window.close()


def test_mainwindow_geometry_point_load_is_re_resolved_after_meshing() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        backend_index = window._mesh_backend_combo.findData("builtin")
        if backend_index >= 0:
            window._mesh_backend_combo.setCurrentIndex(backend_index)
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._sync_scene_from_sketch_geometry()
        point_id = "loop:active:pt:0"
        set_id = "set:point-load"
        window._scene_project.geometry_sets[set_id] = GeometrySetDef(
            id=set_id,
            name="Point Load",
            entity_type="point",
            entity_ids=[point_id],
            binding_mode="geometry",
        )
        window._scene_project.load_definitions["load:1"] = LoadDefinition(
            id="load:1",
            name="Load-1",
            target_set_id=set_id,
            target_entity_type="point",
            load_type="concentrated",
            vector_x=1.0,
            vector_y=0.0,
            magnitude=5.0,
        )

        assert window._run_mesh_backend_operation(operation="generate")

        assert window._model is not None
        assert len(window._model.loads) == 1
        assert window._model.loads[0].node_id in window._scene_mesh_state.point_to_node_ids[point_id]
    finally:
        window.close()
