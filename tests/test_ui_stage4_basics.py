import math
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, QPointF, QSettings
from PySide6.QtWidgets import QApplication

from fem_ai_solver.fem.model import Material
from fem_ai_solver.fem.services import solve_linear_static
from fem_ai_solver.preprocessing.demo_model import build_stage1_demo_model
from fem_ai_solver.ui.colormap import Colormap
from fem_ai_solver.ui.i18n import load_language, normalize_language, save_language, ui_text
from fem_ai_solver.ui.main_window import MainWindow
from fem_ai_solver.ui.mesh_canvas import MeshCanvas, _ScreenTransform
from fem_ai_solver.ui.startup_dialog import StartupDialog


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_i18n_language_settings_roundtrip() -> None:
    runtime_dir = Path("tests/.tmp_runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    settings_path = runtime_dir / "ui_settings_stage4.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)

    save_language("en_US", settings)
    assert load_language(settings) == "en_US"

    save_language("zh_CN", settings)
    assert load_language(settings) == "zh_CN"

    assert normalize_language("invalid") == "zh_CN"
    assert ui_text("toolbar.solve", "zh_CN", "Solve") == "求解"
    assert ui_text("toolbar.solve", "en_US", "Solve") == "Solve"


def test_colormap_outputs_distinct_colors() -> None:
    colormap = Colormap()
    low = colormap.color(0.0, 0.0, 10.0)
    high = colormap.color(10.0, 0.0, 10.0)
    mid = colormap.color(5.0, 0.0, 10.0)

    assert (low.red(), low.green(), low.blue()) != (high.red(), high.green(), high.blue())
    assert (mid.red(), mid.green(), mid.blue()) != (low.red(), low.green(), low.blue())


def test_startup_dialog_choice_and_language() -> None:
    _get_app()
    dialog = StartupDialog("en_US")
    assert "Start" in dialog.windowTitle()

    dialog.retranslate("zh_CN")
    assert "开始" in dialog.windowTitle()

    dialog._accept_with("demo")
    assert dialog.choice.action == "demo"


def test_mainwindow_can_start_empty_without_demo() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        assert window._model is not None
        assert len(window._model.mesh.nodes) == 0
        assert len(window._model.mesh.elements) == 0
    finally:
        window.close()


def test_mesh_canvas_contour_scalar_field_available_after_solve() -> None:
    _get_app()
    model = build_stage1_demo_model()
    result = solve_linear_static(model, backend="python")

    canvas = MeshCanvas()
    canvas.set_model(model)
    canvas.set_result(result)
    canvas.set_contour_variable("sx")
    scalar_bundle = canvas._element_scalar_field()

    assert scalar_bundle is not None
    values, vmin, vmax = scalar_bundle
    assert len(values) == len(model.mesh.elements)
    assert vmax >= vmin


def test_mainwindow_geotech_quick_template_pipeline() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._run_geotech_quick_template()
        assert window._model is not None
        assert window._result is not None
        assert len(window._model.mesh.nodes) > 0
        assert len(window._model.mesh.elements) > 0
        assert "A:" in window._geotech_point_summary.text()
    finally:
        window.close()


def test_mainwindow_geotech_mesh_size_parameter_changes_mesh_density() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._geotech_mesh_size_spin.setValue(2.5)
        window._run_geotech_quick_template()
        assert window._model is not None
        coarse_nodes = len(window._model.mesh.nodes)

        window._geotech_mesh_size_spin.setValue(0.8)
        window._run_geotech_quick_template()
        assert window._model is not None
        fine_nodes = len(window._model.mesh.nodes)

        assert fine_nodes > coarse_nodes
    finally:
        window.close()


def test_mainwindow_editor_can_apply_minimal_t3_model() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._append_editor_row(window._editor_material_table, ["1", "2.0e7", "0.30", "false"])
        window._append_editor_row(window._editor_node_table, ["1", "0.0", "0.0"])
        window._append_editor_row(window._editor_node_table, ["2", "1.0", "0.0"])
        window._append_editor_row(window._editor_node_table, ["3", "0.0", "1.0"])
        window._append_editor_row(window._editor_element_table, ["1", "1", "2", "3", "1"])

        window._apply_editor_draft_to_model()
        assert window._model is not None
        assert len(window._model.mesh.nodes) == 3
        assert len(window._model.mesh.elements) == 1
        assert len(window._model.materials) == 1
    finally:
        window.close()


def test_mainwindow_canvas_edit_handlers_modify_model() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._load_demo_model()
        assert window._model is not None
        initial_nodes = len(window._model.mesh.nodes)
        initial_elements = len(window._model.mesh.elements)

        window._on_canvas_node_created(0.2, 0.3)
        assert window._model is not None
        assert len(window._model.mesh.nodes) == initial_nodes + 1
        new_node_id = max(node.id for node in window._model.mesh.nodes)

        node1 = window._get_node_by_id(1)
        assert node1 is not None
        old_x = node1.x
        old_y = node1.y
        window._on_canvas_node_moved(1, old_x + 0.05, old_y + 0.01)
        moved_node = window._get_node_by_id(1)
        assert moved_node is not None
        assert moved_node.x == old_x + 0.05

        window._editor_canvas_material_spin.setValue(1)
        window._on_canvas_element_created([1, 2, new_node_id])
        assert window._model is not None
        assert len(window._model.mesh.elements) == initial_elements + 1
    finally:
        window.close()


def test_mainwindow_material_creation_syncs_manager_and_assignment_widgets() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._material_name_edit.setText("Clay-1")
        window._material_e_spin.setValue(3.2e7)
        window._material_nu_spin.setValue(0.28)
        window._material_plane_stress.setChecked(False)

        window._create_isotropic_material()

        assert window._model is not None
        assert [material.id for material in window._model.materials] == [1]
        assert window._material_manager_table.rowCount() == 1
        assert window._material_assign_combo.count() == 1
        assert window._material_assign_combo.currentData() == 1
        assert window._material_toolbar_combo.currentData() == 1
        assert "Clay-1" in window._material_manager_table.item(0, 1).text()
    finally:
        window.close()


def test_mainwindow_material_manager_selection_updates_form_and_current_target() -> None:
    app = _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._material_name_edit.setText("Clay-1")
        window._material_e_spin.setValue(3.0e7)
        window._material_nu_spin.setValue(0.27)
        window._create_isotropic_material()

        window._material_name_edit.setText("Rock-2")
        window._material_e_spin.setValue(8.5e7)
        window._material_nu_spin.setValue(0.22)
        window._material_plane_stress.setChecked(True)
        window._create_isotropic_material()

        window._material_manager_table.selectRow(0)
        app.processEvents()

        assert window._material_assign_combo.currentData() == 1
        assert window._material_toolbar_combo.currentData() == 1
        assert window._material_name_edit.text() == "Clay-1"
        assert window._material_e_spin.value() == 3.0e7
        assert window._material_nu_spin.value() == 0.27
    finally:
        window.close()


def test_mainwindow_delete_unused_material_refreshes_manager_and_selection() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._material_name_edit.setText("Clay-1")
        window._create_isotropic_material()

        window._material_name_edit.setText("Rock-2")
        window._material_e_spin.setValue(9.1e7)
        window._create_isotropic_material()

        assert window._material_assign_combo.currentData() == 2

        window._delete_selected_material()

        assert window._model is not None
        assert [material.id for material in window._model.materials] == [1]
        assert window._material_manager_table.rowCount() == 1
        assert window._material_assign_combo.count() == 1
        assert window._material_assign_combo.currentData() == 1
        assert window._material_toolbar_combo.currentData() == 1
    finally:
        window.close()


def test_mainwindow_material_workflow_shows_material_toolbar() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._activate_workflow("material")
        assert not window._material_tool_strip.isHidden()
        assert window._material_toolbar_template_combo.count() >= 1
        assert window._btn_material_tool_rename.toolTip().strip() != ""
        assert window._btn_material_tool_duplicate.toolTip().strip() != ""

        window._activate_workflow("part")
        assert window._material_tool_strip.isHidden()
    finally:
        window.close()


def test_mainwindow_material_assignment_toolbar_applies_geotech_template() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        assert window._material_action_template_combo.count() >= 1
        weathered_index = window._material_action_template_combo.findData("weathered_rock")
        assert weathered_index >= 0

        window._material_action_template_combo.setCurrentIndex(weathered_index)
        window._apply_selected_material_template()

        assert abs(window._material_e_spin.value() - 8.0e8) < 1e-3
        assert abs(window._material_nu_spin.value() - 0.26) < 1e-9
        assert not window._material_plane_stress.isChecked()

        window._material_name_edit.setText("Weathered-Rock")
        window._create_isotropic_material()

        assert window._model is not None
        assert window._model.materials[0].id == 1
        assert window._material_manager_table.rowCount() == 1
        assert window._material_assign_combo.currentData() == 1
        assert window._btn_material_action_rename.isEnabled()
        assert window._btn_material_action_duplicate.isEnabled()
        assert window._btn_material_action_assign.isEnabled()
    finally:
        window.close()


def test_mainwindow_material_rename_and_duplicate_refresh_manager_widgets() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._material_name_edit.setText("Clay-1")
        window._material_e_spin.setValue(3.0e7)
        window._material_nu_spin.setValue(0.27)
        window._create_isotropic_material()

        with patch("fem_ai_solver.ui.main_window.QInputDialog.getText", return_value=("Clay-A", True)):
            window._rename_selected_material()

        assert window._material_display_name(1) == "Clay-A"
        assert "Clay-A" in window._material_manager_table.item(0, 1).text()
        assert window._material_assign_combo.itemText(0).startswith("Clay-A")

        with patch("fem_ai_solver.ui.main_window.QInputDialog.getText", return_value=("Clay-A Copy", True)):
            window._duplicate_selected_material()

        assert window._model is not None
        assert [int(material.id) for material in window._model.materials] == [1, 2]
        duplicate = next(material for material in window._model.materials if int(material.id) == 2)
        assert abs(float(duplicate.young_modulus) - 3.0e7) < 1e-6
        assert abs(float(duplicate.poisson_ratio) - 0.27) < 1e-9
        assert window._material_manager_table.rowCount() == 2
        assert window._material_assign_combo.currentData() == 2
        assert window._material_toolbar_combo.currentData() == 2
        assert window._material_display_name(2) == "Clay-A Copy"
    finally:
        window.close()


def test_mainwindow_material_selection_highlights_usage_and_bulk_reassigns() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._load_demo_model()
        assert window._model is not None
        base = window._model.materials[0]
        window._model.materials.append(
            Material(
                id=2,
                young_modulus=base.young_modulus,
                poisson_ratio=base.poisson_ratio,
                plane_stress=base.plane_stress,
            )
        )
        highlighted = set()
        for idx, element in enumerate(window._model.mesh.elements):
            if idx % 2 == 0:
                element.material_id = 2
                highlighted.add(int(element.id))
        window._set_model(window._model)
        window._material_id_to_name[2] = "Rock-2"
        window._material_name_to_id = {name: mid for mid, name in window._material_id_to_name.items()}
        window._sync_material_controls()

        window._select_material_by_id(2)

        assert window._mesh_canvas._highlighted_element_ids == highlighted
        usage_text = window._material_usage_details.toPlainText()
        assert "Rock-2" in usage_text
        assert "Element coverage" in usage_text

        target_index = window._material_reassign_target_combo.findData(1)
        assert target_index >= 0
        window._material_reassign_target_combo.setCurrentIndex(target_index)
        window._batch_reassign_selected_material()

        assert all(int(element.material_id) == 1 for element in window._model.mesh.elements)
        assert window._material_assign_combo.currentData() == 1
    finally:
        window.close()


def test_mainwindow_overlapping_sketch_regions_are_split_into_atomic_faces() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._part_sketch_points = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
        window._part_sketch_history = [
            {"points": [], "hint": {"kind": "circle", "bbox": (2.0, -0.5, 5.0, 2.5)}}
        ]
        window._refresh_part_sketch_table()

        regions = window._collect_sketch_face_regions()
        keys = {str(item["key"]) for item in regions}

        assert len(regions) == 3
        assert any(key.startswith("split:active:") for key in keys)
        assert any(key.startswith("split:history:0:") for key in keys)
        assert any(key.startswith("overlap:") for key in keys)
        overlap = window._pick_sketch_face_region_at(3.0, 1.0)
        assert overlap is not None
        assert str(overlap["key"]).startswith("overlap:")
    finally:
        window.close()


def test_mainwindow_material_sketch_pick_mode_supports_continuous_multi_select() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._material_name_edit.setText("Soil-1")
        window._create_isotropic_material()
        window._part_sketch_points = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0), (0.0, 0.0)]
        window._part_sketch_history = [
            {"points": [], "hint": {"kind": "circle", "bbox": (2.0, -0.5, 5.0, 2.5)}}
        ]
        window._refresh_part_sketch_table()

        window._start_pick_face_region()
        assert window._pending_pick_context == "material_face_sketch"

        window._on_canvas_sketch_face_picked(0.5, 1.0)
        assert len(window._selected_face_region_sketch_keys) == 1
        assert window._pending_pick_context == "material_face_sketch"

        window._on_canvas_sketch_face_picked(3.0, 1.0)
        assert len(window._selected_face_region_sketch_keys) == 2
        assert window._pending_pick_context == "material_face_sketch"
        assert len(window._mesh_canvas._highlighted_sketch_faces) == 2

        selected_region_ids = {
            window._scene_region_id_from_key(key)
            for key in window._selected_face_region_sketch_keys
        }
        window._assign_material_to_region()

        assert all(window._scene_project.regions[region_id].material_id == 1 for region_id in selected_region_ids)
        assert "Sketch-region bindings: 2" in window._material_usage_details.toPlainText()
    finally:
        window.close()


def test_mainwindow_sketch_pick_overlay_and_overlap_region_survive_geometry_refresh() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._material_name_edit.setText("Soil-1")
        window._create_isotropic_material()

        window._on_canvas_sketch_primitive_drawn("rect", 0.0, 0.0, 3.0, 3.0)
        window._start_pick_face_region()
        assert window._pending_pick_context == "material_face_sketch"
        assert len(window._mesh_canvas._pickable_sketch_faces) == 1

        window._on_canvas_sketch_primitive_drawn("circle", 1.8, -1.2, 4.8, 1.8)
        window._start_pick_face_region()
        window._start_pick_face_region()

        assert window._pending_pick_context == "material_face_sketch"
        assert len(window._mesh_canvas._pickable_sketch_faces) == 3

        overlap = window._pick_sketch_face_region_at(2.4, 0.8)
        assert overlap is not None
        assert str(overlap["key"]).startswith("overlap:")

        window._on_canvas_sketch_face_picked(2.4, 0.8)
        assert len(window._selected_face_region_sketch_keys) == 1
        picked_key = next(iter(window._selected_face_region_sketch_keys))
        assert picked_key.startswith("overlap:")
    finally:
        window.close()


def test_mainwindow_escape_cancel_clears_sketch_capture_and_preview() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points.extend([(0.0, 0.0), (1.0, 0.0)])
        window._refresh_part_sketch_table()
        window._mesh_canvas._sketch_hover = (0.5, 0.4)
        window._mesh_canvas._pending_element_nodes = [1, 2]
        window._pending_pick_context = "material_face"
        add_node_index = window._editor_canvas_tool_combo.findData("add_node")
        assert add_node_index >= 0
        window._editor_canvas_tool_combo.setCurrentIndex(add_node_index)
        assert window._selected_canvas_tool() == "add_node"
        assert window._capture_sketch_from_canvas
        assert window._part_capture_from_canvas.isChecked()

        window._cancel_interactive_modes()

        assert not window._capture_sketch_from_canvas
        assert not window._part_capture_from_canvas.isChecked()
        assert window._pending_pick_context is None
        assert window._selected_canvas_tool() == "select"
        assert window._mesh_canvas._sketch_hover is None
        assert window._mesh_canvas._pending_element_nodes == []
    finally:
        window.close()


def test_mainwindow_part_sketch_undo_redo_flow() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_point_x.setValue(0.0)
        window._part_point_y.setValue(0.0)
        window._add_part_sketch_point()
        window._part_point_x.setValue(1.0)
        window._part_point_y.setValue(0.0)
        window._add_part_sketch_point()
        window._part_point_x.setValue(1.0)
        window._part_point_y.setValue(1.0)
        window._add_part_sketch_point()
        assert len(window._part_sketch_points) == 3
        assert window._part_sketch_redo_stack == []
        assert len(window._part_sketch_undo_stack) >= 1

        window._undo_last_sketch_point()
        assert len(window._part_sketch_points) == 2
        assert len(window._part_sketch_redo_stack) == 1

        window._redo_last_sketch_point()
        assert len(window._part_sketch_points) == 3
        assert window._part_sketch_redo_stack == []

        window._undo_last_sketch_point()
        assert len(window._part_sketch_redo_stack) == 1
        window._part_point_x.setValue(2.0)
        window._part_point_y.setValue(1.0)
        window._add_part_sketch_point()
        assert window._part_sketch_redo_stack == []
    finally:
        window.close()


def test_mainwindow_part_undo_redo_restores_dimension_edit() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._part_sketch_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        window._refresh_part_sketch_table()
        window._on_canvas_sketch_segment_picked(0, 1)
        window._part_length_target.setValue(2.0)
        window._apply_part_length_constraint()
        assert abs(window._part_sketch_points[1][0] - 2.0) < 1e-8

        window._undo_last_sketch_point()
        assert abs(window._part_sketch_points[1][0] - 1.0) < 1e-8

        window._redo_last_sketch_point()
        assert abs(window._part_sketch_points[1][0] - 2.0) < 1e-8
    finally:
        window.close()


def test_mesh_canvas_zoom_controls_change_zoom_level() -> None:
    _get_app()
    canvas = MeshCanvas()
    base_zoom = canvas._view_zoom
    canvas.zoom_in()
    assert canvas._view_zoom > base_zoom
    zoom_after_in = canvas._view_zoom
    canvas.zoom_out()
    assert canvas._view_zoom < zoom_after_in


def test_mainwindow_part_dimension_tools_apply_length_and_angle() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._part_sketch_points = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]
        window._refresh_part_sketch_table()
        table = window._part_sketch_table
        sel = table.selectionModel()
        assert sel is not None

        sel.select(table.model().index(0, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        sel.select(table.model().index(1, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        window._part_length_target.setValue(2.0)
        window._apply_part_length_constraint()
        p0 = window._part_sketch_points[0]
        p1 = window._part_sketch_points[1]
        distance = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        assert abs(distance - 2.0) < 1e-8

        sel.clearSelection()
        sel.select(table.model().index(0, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        sel.select(table.model().index(1, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        sel.select(table.model().index(2, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        window._part_angle_target.setValue(90.0)
        window._apply_part_angle_constraint()
        pa, pb, pc = window._part_sketch_points[0], window._part_sketch_points[1], window._part_sketch_points[2]
        v1 = (pa[0] - pb[0], pa[1] - pb[1])
        v2 = (pc[0] - pb[0], pc[1] - pb[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        n1 = (v1[0] * v1[0] + v1[1] * v1[1]) ** 0.5
        n2 = (v2[0] * v2[0] + v2[1] * v2[1]) ** 0.5
        cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))
        angle_deg = math.degrees(math.acos(cos_theta))
        assert abs(angle_deg - 90.0) < 1e-5
    finally:
        window.close()


def test_mainwindow_icon_buttons_have_tooltips() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        assert window._btn_quick_undo.toolTip().strip() != ""
        assert window._btn_quick_zoom_in.toolTip().strip() != ""
        assert window._btn_part_tool_clear_selected.toolTip().strip() != ""
        assert window._btn_part_tool_clear_all.toolTip().strip() != ""
        assert window._btn_toolbar_solve.toolTip().strip() != ""
        assert window._workflow_buttons["part"].toolTip().strip() != ""
    finally:
        window.close()


def test_mainwindow_part_strip_visibility_and_quick_dimension_apply() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._activate_workflow("model")
        assert window._part_sketch_strip.isHidden()
        window._activate_workflow("part")
        assert not window._part_sketch_strip.isHidden()

        window._part_sketch_points = [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)]
        window._refresh_part_sketch_table()
        table = window._part_sketch_table
        sel = table.selectionModel()
        assert sel is not None

        sel.select(table.model().index(0, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        sel.select(table.model().index(1, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        window._part_quick_mode.setCurrentIndex(window._part_quick_mode.findData("length"))
        window._part_quick_value.setValue(3.0)
        window._apply_quick_dimension_value()
        p0, p1 = window._part_sketch_points[0], window._part_sketch_points[1]
        dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        assert abs(dist - 3.0) < 1e-8

        sel.clearSelection()
        sel.select(table.model().index(0, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        sel.select(table.model().index(1, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        sel.select(table.model().index(2, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        window._part_quick_mode.setCurrentIndex(window._part_quick_mode.findData("angle"))
        window._part_quick_value.setValue(60.0)
        window._apply_quick_dimension_value()
        pa, pb, pc = window._part_sketch_points[0], window._part_sketch_points[1], window._part_sketch_points[2]
        v1 = (pa[0] - pb[0], pa[1] - pb[1])
        v2 = (pc[0] - pb[0], pc[1] - pb[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        n1 = (v1[0] * v1[0] + v1[1] * v1[1]) ** 0.5
        n2 = (v2[0] * v2[0] + v2[1] * v2[1]) ** 0.5
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))
        assert abs(angle_deg - 60.0) < 1e-5
    finally:
        window.close()


def test_mesh_canvas_measure_angle_emits_angle() -> None:
    _get_app()
    canvas = MeshCanvas()
    seen: list[float] = []
    canvas.measurement_angle_updated.connect(lambda value: seen.append(float(value)))
    canvas.set_edit_tool("measure_angle")
    canvas._handle_measure_pick(0.0, 0.0)
    canvas._handle_measure_pick(0.0, 1.0)
    canvas._handle_measure_pick(1.0, 1.0)
    assert seen
    assert abs(seen[-1] - 90.0) < 1e-6


def test_mesh_canvas_pick_sketch_angle_supports_click_on_angle_arm() -> None:
    _get_app()
    canvas = MeshCanvas()
    canvas.set_sketch_points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    transform = _ScreenTransform(
        min_x=0.0,
        min_y=0.0,
        scale=100.0,
        height=260.0,
        padding=20.0,
        zoom=1.0,
        pan_x=0.0,
        pan_y=0.0,
    )
    # Click near the vertical arm from vertex (1,0) to (1,1).
    click = transform.map(1.0, 0.42)
    triple = canvas._pick_sketch_angle(click, transform)
    assert triple == (0, 1, 2)


def test_mesh_canvas_pick_sketch_angle_rejects_far_click() -> None:
    _get_app()
    canvas = MeshCanvas()
    canvas.set_sketch_points([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    transform = _ScreenTransform(
        min_x=0.0,
        min_y=0.0,
        scale=100.0,
        height=260.0,
        padding=20.0,
        zoom=1.0,
        pan_x=0.0,
        pan_y=0.0,
    )
    far = QPointF(999.0, 999.0)
    assert canvas._pick_sketch_angle(far, transform) is None


def test_mesh_canvas_measure_snap_includes_circle_center() -> None:
    _get_app()
    canvas = MeshCanvas()
    seen: list[float] = []
    canvas.measurement_updated.connect(lambda value: seen.append(float(value)))
    canvas.set_edit_tool("measure")
    canvas.set_sketch_points([(0.0, 0.0)])
    canvas.set_sketch_curve_hint({"kind": "circle", "bbox": (0.0, 0.0, 2.0, 2.0)})
    transform = _ScreenTransform(
        min_x=0.0,
        min_y=0.0,
        scale=100.0,
        height=260.0,
        padding=20.0,
        zoom=1.0,
        pan_x=0.0,
        pan_y=0.0,
    )

    assert canvas._handle_measure_snap_pick(transform.map(0.0, 0.0), transform)
    assert canvas._handle_measure_snap_pick(transform.map(1.0, 1.0), transform)

    assert seen
    assert abs(seen[-1] - math.sqrt(2.0)) < 1e-9


def test_mesh_canvas_angle_segment_order_composition() -> None:
    # First selected edge: (0-1), second selected edge: (1-2) -> triple should be (0,1,2)
    assert MeshCanvas._compose_angle_from_segment_order((0, 1), (1, 2)) == (0, 1, 2)
    # Reverse order means reverse rotating side.
    assert MeshCanvas._compose_angle_from_segment_order((2, 1), (1, 0)) == (2, 1, 0)


def test_mainwindow_canvas_picked_segment_can_drive_length_update() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._part_sketch_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        window._refresh_part_sketch_table()
        window._on_canvas_sketch_segment_picked(0, 1)
        assert window._selected_part_segment == (0, 1)
        window._part_quick_mode.setCurrentIndex(window._part_quick_mode.findData("length"))
        window._part_quick_value.setValue(2.5)
        window._apply_quick_dimension_value()
        p0, p1 = window._part_sketch_points[0], window._part_sketch_points[1]
        dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        assert abs(dist - 2.5) < 1e-8
    finally:
        window.close()


def test_mainwindow_canvas_picked_segment_order_controls_moving_side() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._part_sketch_points = [(0.0, 0.0), (1.0, 0.0)]
        window._refresh_part_sketch_table()
        # Reversed order means node 1 fixed and node 0 moved.
        window._on_canvas_sketch_segment_picked(1, 0)
        window._part_quick_mode.setCurrentIndex(window._part_quick_mode.findData("length"))
        window._part_quick_value.setValue(3.0)
        window._apply_quick_dimension_value()
        assert abs(window._part_sketch_points[1][0] - 1.0) < 1e-10
        assert abs(window._part_sketch_points[0][0] + 2.0) < 1e-8
    finally:
        window.close()


def test_mainwindow_canvas_picked_angle_can_drive_angle_update() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
        window._refresh_part_sketch_table()
        window._on_canvas_sketch_angle_picked(0, 1, 2)
        assert window._selected_part_angle == (0, 1, 2)
        window._part_quick_mode.setCurrentIndex(window._part_quick_mode.findData("angle"))
        window._part_quick_value.setValue(45.0)
        window._apply_quick_dimension_value()
        pa, pb, pc = window._part_sketch_points[0], window._part_sketch_points[1], window._part_sketch_points[2]
        v1 = (pa[0] - pb[0], pa[1] - pb[1])
        v2 = (pc[0] - pb[0], pc[1] - pb[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        n1 = (v1[0] * v1[0] + v1[1] * v1[1]) ** 0.5
        n2 = (v2[0] * v2[0] + v2[1] * v2[1]) ** 0.5
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))
        assert abs(angle_deg - 45.0) < 1e-5
    finally:
        window.close()


def test_mainwindow_canvas_picked_angle_order_controls_rotating_side() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
        window._refresh_part_sketch_table()
        # Reverse side order: keep edge (2-1) as base, rotate side (1-0), so point 0 should move while point 2 stays.
        window._on_canvas_sketch_angle_picked(2, 1, 0)
        p2_before = window._part_sketch_points[2]
        p0_before = window._part_sketch_points[0]
        window._part_quick_mode.setCurrentIndex(window._part_quick_mode.findData("angle"))
        window._part_quick_value.setValue(60.0)
        window._apply_quick_dimension_value()
        p2_after = window._part_sketch_points[2]
        p0_after = window._part_sketch_points[0]
        assert abs(p2_after[0] - p2_before[0]) < 1e-10
        assert abs(p2_after[1] - p2_before[1]) < 1e-10
        assert abs(p0_after[0] - p0_before[0]) > 1e-6 or abs(p0_after[1] - p0_before[1]) > 1e-6
    finally:
        window.close()


def test_mainwindow_insert_ellipse_uses_custom_size_controls() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_shape_w.setValue(4.0)
        window._part_shape_h.setValue(2.0)
        window._insert_part_ellipse()
        xs = [point[0] for point in window._part_sketch_points[:-1]]
        ys = [point[1] for point in window._part_sketch_points[:-1]]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        assert abs(width - 4.0) < 1e-2
        assert abs(height - 2.0) < 1e-2
    finally:
        window.close()


def test_mainwindow_canvas_primitive_drawn_rect_respects_drag_bbox() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._on_canvas_sketch_primitive_drawn("rect", 1.0, 2.0, 5.5, 4.5)
        pts = window._part_sketch_points
        assert pts[0] == (1.0, 2.0)
        assert pts[1] == (5.5, 2.0)
        assert pts[2] == (5.5, 4.5)
        assert pts[3] == (1.0, 4.5)
    finally:
        window.close()


def test_mainwindow_canvas_primitive_drawn_ellipse_updates_shape_controls() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._on_canvas_sketch_primitive_drawn("ellipse", 2.0, 3.0, 8.0, 7.0)
        assert abs(window._part_shape_w.value() - 6.0) < 1e-9
        assert abs(window._part_shape_h.value() - 4.0) < 1e-9
        assert len(window._part_sketch_points) > 10
        assert window._part_sketch_points[0] == window._part_sketch_points[-1]
        assert window._part_sketch_curve_hint is not None
        assert window._part_sketch_curve_hint.get("kind") == "ellipse"
    finally:
        window.close()


def test_mainwindow_canvas_primitive_drawn_archives_previous_geometry() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._on_canvas_sketch_primitive_drawn("rect", 0.0, 0.0, 2.0, 1.0)
        assert len(window._part_sketch_history) == 0
        window._on_canvas_sketch_primitive_drawn("rect", 3.0, 3.0, 5.0, 5.0)
        assert len(window._part_sketch_history) == 1
        old_points = window._part_sketch_history[0].get("points")
        assert isinstance(old_points, list)
        assert old_points[0] == (0.0, 0.0)
        assert old_points[1] == (2.0, 0.0)
        assert old_points[2] == (2.0, 1.0)
    finally:
        window.close()


def test_mainwindow_clear_selected_and_clear_all_part_geometry() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        window._refresh_part_sketch_table()
        table = window._part_sketch_table
        sel = table.selectionModel()
        idx = table.model().index(1, 0)
        sel.select(idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        window._clear_selected_part_geometry()
        assert len(window._part_sketch_points) == 2

        window._part_sketch_history = [{"points": [(2.0, 2.0), (3.0, 2.0)], "hint": None}]
        window._clear_all_part_geometry()
        assert window._part_sketch_points == []
        assert window._part_sketch_history == []
    finally:
        window.close()


def test_mainwindow_history_segment_pick_activates_shape_for_dimension_edit() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._on_canvas_sketch_primitive_drawn("rect", 0.0, 0.0, 2.0, 1.0)
        window._on_canvas_sketch_primitive_drawn("rect", 4.0, 0.0, 6.0, 1.0)
        assert len(window._part_sketch_history) == 1
        assert window._part_sketch_points[0] == (4.0, 0.0)

        window._on_canvas_sketch_history_segment_picked(0, 0, 1)
        assert window._part_sketch_points[0] == (0.0, 0.0)
        assert window._selected_part_segment == (0, 1)
        assert len(window._part_sketch_history) == 1

        window._part_quick_mode.setCurrentIndex(window._part_quick_mode.findData("length"))
        window._part_quick_value.setValue(3.0)
        window._apply_quick_dimension_value()
        p0, p1 = window._part_sketch_points[0], window._part_sketch_points[1]
        dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        assert abs(dist - 3.0) < 1e-6
    finally:
        window.close()


def test_mesh_canvas_can_pick_history_segment() -> None:
    _get_app()
    canvas = MeshCanvas()
    canvas.set_sketch_plane(10.0, 10.0, 1.0)
    canvas.set_sketch_points([])
    canvas.set_sketch_history(
        [
            {
                "points": [(1.0, 1.0), (4.0, 1.0), (4.0, 3.0)],
                "hint": None,
            }
        ]
    )
    transform = _ScreenTransform(min_x=0.0, min_y=0.0, scale=40.0, height=400.0, padding=20.0, zoom=1.0, pan_x=0.0, pan_y=0.0)
    click = transform.map(2.5, 1.05)
    hit = canvas._pick_history_segment(click, transform)
    assert hit is not None
    h_idx, i0, i1 = hit
    assert h_idx == 0
    assert (i0, i1) == (0, 1)


def test_mainwindow_part_constraint_buttons_have_labels() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        assert window._btn_part_clear_selected.text().strip() != ""
        assert window._btn_part_constraint_horizontal.text().strip() != ""
        assert window._btn_part_constraint_vertical.text().strip() != ""
        assert window._btn_part_constraint_collinear.text().strip() != ""
        assert window._btn_part_constraint_parallel.text().strip() != ""
        assert window._btn_part_constraint_perpendicular.text().strip() != ""
    finally:
        window.close()


def test_mainwindow_results_scope_filter_by_material_limits_element_table_and_canvas() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._load_demo_model()
        assert window._model is not None
        if not any(int(item.id) == 2 for item in window._model.materials):
            base = window._model.materials[0]
            window._model.materials.append(
                Material(
                    id=2,
                    young_modulus=base.young_modulus,
                    poisson_ratio=base.poisson_ratio,
                    plane_stress=base.plane_stress,
                )
            )
        for idx, element in enumerate(window._model.mesh.elements):
            if idx % 2 == 0:
                element.material_id = 2
        window._set_model(window._model)
        window._solve()

        type_idx = window._results_scope_type_combo.findData("material")
        assert type_idx >= 0
        window._results_scope_type_combo.setCurrentIndex(type_idx)
        target_idx = window._results_scope_target_combo.findData("2")
        assert target_idx >= 0
        window._results_scope_target_combo.setCurrentIndex(target_idx)

        visible_element_ids = {
            int(window._element_table.item(row, 0).text())
            for row in range(window._element_table.rowCount())
        }
        assert visible_element_ids
        assert all(
            int(element.material_id) == 2
            for element in window._model.mesh.elements
            if int(element.id) in visible_element_ids
        )
        assert window._mesh_canvas._visible_element_ids is not None
        assert visible_element_ids.issubset(window._mesh_canvas._visible_element_ids)
    finally:
        window.close()


def test_mainwindow_can_capture_region_and_component_sets() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()
        assert "region:active" in window._scene_project.regions
        window._selected_face_region_sketch_keys = {"active"}
        window._capture_region_set_from_selected_regions()
        assert any(
            item.entity_type == "region" and "region:active" in item.entity_ids
            for item in window._scene_project.geometry_sets.values()
        )

        window._load_demo_model()
        if window._component_manager_table.rowCount() > 0:
            window._component_manager_table.selectRow(0)
        window._capture_component_set_from_selected_component()
        assert any(
            item.entity_type == "component" and len(item.entity_ids) >= 1
            for item in window._scene_project.geometry_sets.values()
        )
    finally:
        window.close()


def test_mainwindow_sketch_sync_builds_geometry_points_and_edges() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()
        sketch_points = [item for item in window._scene_project.geometry_points.values() if item.source == "sketch"]
        sketch_edges = [item for item in window._scene_project.geometry_edges.values() if item.source == "sketch"]
        assert len(sketch_points) == 4
        assert len(sketch_edges) == 4
    finally:
        window.close()


def test_mainwindow_geometry_point_target_set_resolves_to_mesh_nodes() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        point_id = window._resolve_or_create_geometry_point_at(0.0, 0.0)
        assert point_id is not None
        window._selected_geometry_point_ids = {point_id}
        window._capture_geometry_set_from_current_target()

        set_ids = [
            set_id
            for set_id, item in window._scene_project.geometry_sets.items()
            if item.entity_type == "point" and item.binding_mode == "geometry"
        ]
        assert set_ids
        target_set_id = set_ids[-1]

        window._build_part_from_sketch()
        node_ids = window._scene_mesh_state.set_to_node_ids.get(target_set_id, [])
        assert len(node_ids) == 1
    finally:
        window.close()


def test_mainwindow_interior_geometry_point_triggers_constrained_meshing_and_maps() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        point_id = window._resolve_or_create_geometry_point_at(1.0, 0.5)
        assert point_id is not None
        assert window._scene_project.geometry_points[point_id].role == "interior_point"
        assert window._requires_constrained_geometry_mesh()

        window._build_part_from_sketch()

        mapped_nodes = window._scene_mesh_state.point_to_node_ids.get(point_id, [])
        assert mapped_nodes
    finally:
        window.close()


def test_mainwindow_geometry_point_load_saves_then_resolves_after_meshing() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        point_id = window._resolve_or_create_geometry_point_at(0.73, 0.41)
        assert point_id is not None
        assert window._scene_project.geometry_points[point_id].role == "interior_point"
        window._selected_geometry_point_ids = {point_id}
        window._capture_geometry_set_from_current_target()

        set_ids = [
            set_id
            for set_id, item in window._scene_project.geometry_sets.items()
            if item.entity_type == "point" and item.binding_mode == "geometry"
        ]
        assert set_ids
        target_set_id = set_ids[-1]
        idx = window._load_bc_target_set_combo.findData(target_set_id)
        assert idx >= 0
        window._load_bc_target_set_combo.setCurrentIndex(idx)

        # Before auto-remesh, the interior point is typically unresolved on current mesh.
        assert window._scene_mesh_state.set_to_node_ids.get(target_set_id, []) == []

        window._load_point_magnitude_spin.setValue(10.0)
        window._load_point_vec_x.setValue(1.0)
        window._load_point_vec_y.setValue(0.0)
        window._add_concentrated_load()

        assert window._scene_project.load_definitions
        assert window._scene_mesh_state.set_to_node_ids.get(target_set_id, []) == []
        assert window._model.loads == []

        window._build_part_from_sketch()

        assert window._scene_mesh_state.set_to_node_ids.get(target_set_id, [])
        assert any(load.dof == "fx" for load in window._model.loads)
    finally:
        window.close()


def test_mainwindow_geometry_edge_distributed_load_preserves_total_force_after_meshing() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        edge_id, _ = window._pick_geometry_edge_id_at(1.0, 1.0, tolerance=0.2)
        assert edge_id is not None
        set_id = window._upsert_geometry_set_from_geometry_ids(
            entity_type="edge",
            geometry_ids=[edge_id],
            preferred_name="Top Edge Load Set",
        )
        assert set_id is not None
        idx = window._load_bc_target_set_combo.findData(set_id)
        assert idx >= 0
        window._load_bc_target_set_combo.setCurrentIndex(idx)

        window._load_dist_total_mag_spin.setValue(123.0)
        window._load_dist_vec_x.setValue(0.0)
        window._load_dist_vec_y.setValue(-1.0)
        window._add_distributed_load()

        assert window._scene_project.load_definitions
        assert window._model.loads == []

        window._build_part_from_sketch()

        total_fx = sum(float(load.value) for load in window._model.loads if load.dof == "fx")
        total_fy = sum(float(load.value) for load in window._model.loads if load.dof == "fy")
        assert abs(total_fx) <= 1e-9
        assert abs(total_fy + 123.0) <= 1e-9
    finally:
        window.close()


def test_mainwindow_geometry_edge_pressure_uses_edge_normal_after_meshing() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        edge_id, _ = window._pick_geometry_edge_id_at(1.0, 0.0, tolerance=0.2)
        assert edge_id is not None
        set_id = window._upsert_geometry_set_from_geometry_ids(
            entity_type="edge",
            geometry_ids=[edge_id],
            preferred_name="Bottom Pressure Set",
        )
        assert set_id is not None
        idx = window._load_bc_target_set_combo.findData(set_id)
        assert idx >= 0
        window._load_bc_target_set_combo.setCurrentIndex(idx)

        type_idx = window._load_dist_type_combo.findData("pressure")
        dir_idx = window._load_dist_direction_mode_combo.findData("normal")
        assert type_idx >= 0 and dir_idx >= 0
        window._load_dist_type_combo.setCurrentIndex(type_idx)
        window._load_dist_direction_mode_combo.setCurrentIndex(dir_idx)
        window._load_dist_total_mag_spin.setValue(5.0)
        window._add_distributed_load()

        assert window._scene_project.load_definitions
        assert next(iter(window._scene_project.load_definitions.values())).load_type == "pressure"

        window._build_part_from_sketch()

        total_fx = sum(float(load.value) for load in window._model.loads if load.dof == "fx")
        total_fy = sum(float(load.value) for load in window._model.loads if load.dof == "fy")
        assert abs(total_fx) <= 1e-9
        assert abs(total_fy + 10.0) <= 1e-9
    finally:
        window.close()


def test_mainwindow_region_body_force_resolves_by_area_after_meshing() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        region_id = next(iter(window._scene_project.regions))
        set_id = window._upsert_geometry_set_from_entities(
            entity_type="region",
            entity_ids=[region_id],
            preferred_name="Whole Region",
        )
        assert set_id is not None
        idx = window._load_bc_target_set_combo.findData(set_id)
        assert idx >= 0
        window._load_bc_target_set_combo.setCurrentIndex(idx)

        window._region_load_vec_x.setValue(0.0)
        window._region_load_vec_y.setValue(-1.0)
        window._region_load_intensity_spin.setValue(4.0)
        window._add_region_body_force()

        assert window._scene_project.load_definitions
        assert window._model.loads == []

        window._build_part_from_sketch()

        total_fx = sum(float(load.value) for load in window._model.loads if load.dof == "fx")
        total_fy = sum(float(load.value) for load in window._model.loads if load.dof == "fy")
        assert abs(total_fx) <= 1e-9
        assert abs(total_fy + 8.0) <= 1e-9
    finally:
        window.close()


def test_mainwindow_load_manager_delete_removes_geometry_load_expansion() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        point_id = window._resolve_or_create_geometry_point_at(0.0, 0.0)
        assert point_id is not None
        set_id = window._upsert_geometry_set_from_geometry_ids(
            entity_type="point",
            geometry_ids=[point_id],
            preferred_name="Corner Load",
        )
        assert set_id is not None
        idx = window._load_bc_target_set_combo.findData(set_id)
        assert idx >= 0
        window._load_bc_target_set_combo.setCurrentIndex(idx)

        window._load_point_vec_x.setValue(1.0)
        window._load_point_vec_y.setValue(0.0)
        window._load_point_magnitude_spin.setValue(3.0)
        window._add_concentrated_load()
        window._build_part_from_sketch()

        assert window._scene_project.load_definitions
        assert window._model.loads
        assert window._load_manager_table.rowCount() == 1

        window._load_manager_table.selectRow(0)
        window._delete_selected_load_definition()

        assert not window._scene_project.load_definitions
        assert window._model.loads == []
    finally:
        window.close()


def test_mainwindow_pick_distributed_edge_enters_geometry_edge_context() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mesh_idx = window._load_bc_target_mode_combo.findData("mesh")
        assert mesh_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mesh_idx)

        window._start_pick_distributed_edge_target()

        assert window._load_bc_target_mode_combo.currentData() == "geometry"
        assert window._pending_pick_context == "geometry_edge_target"
        assert window._mesh_canvas.current_edit_tool() == "pick_geometry"
    finally:
        window.close()


def test_mainwindow_distributed_load_on_point_set_guides_to_edge_pick() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        point_id = window._resolve_or_create_geometry_point_at(0.0, 0.0)
        assert point_id is not None
        set_id = window._upsert_geometry_set_from_geometry_ids(
            entity_type="point",
            geometry_ids=[point_id],
            preferred_name="Point Set",
        )
        assert set_id is not None
        idx = window._load_bc_target_set_combo.findData(set_id)
        assert idx >= 0
        window._load_bc_target_set_combo.setCurrentIndex(idx)

        window._add_distributed_load()

        assert window._pending_pick_context == "geometry_edge_target"
        assert not window._scene_project.load_definitions
    finally:
        window.close()


def test_mainwindow_geometry_edge_boundary_condition_resolves_after_meshing() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        edge_id, _ = window._pick_geometry_edge_id_at(1.0, 0.0, tolerance=0.2)
        assert edge_id is not None
        set_id = window._upsert_geometry_set_from_geometry_ids(
            entity_type="edge",
            geometry_ids=[edge_id],
            preferred_name="Bottom Edge BC Set",
        )
        assert set_id is not None
        idx = window._load_bc_target_set_combo.findData(set_id)
        assert idx >= 0
        window._load_bc_target_set_combo.setCurrentIndex(idx)

        dir_idx = window._bc_dir_combo.findData("y")
        assert dir_idx >= 0
        window._bc_dir_combo.setCurrentIndex(dir_idx)
        window._bc_value_spin.setValue(0.0)
        window._apply_constraint()

        assert window._scene_project.boundary_definitions
        assert window._model.boundary_conditions == []

        window._build_part_from_sketch()

        resolved_nodes = set(window._scene_mesh_state.set_to_node_ids.get(set_id, []))
        constrained_nodes = {
            int(bc.node_id)
            for bc in window._model.boundary_conditions
            if bc.dof == "uy" and abs(float(bc.value)) <= 1e-12
        }
        assert resolved_nodes
        assert resolved_nodes.issubset(constrained_nodes)
    finally:
        window.close()


def test_mainwindow_pick_bc_edge_enters_geometry_edge_context() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mesh_idx = window._load_bc_target_mode_combo.findData("mesh")
        assert mesh_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mesh_idx)

        window._start_pick_bc_edge()

        assert window._load_bc_target_mode_combo.currentData() == "geometry"
        assert window._bc_target_combo.currentData() == "edge"
        assert window._pending_pick_context == "geometry_edge_target"
    finally:
        window.close()


def test_mainwindow_bc_fixed_preset_creates_xy_edge_definition() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        edge_id, _ = window._pick_geometry_edge_id_at(1.0, 0.0, tolerance=0.2)
        assert edge_id is not None
        set_id = window._upsert_geometry_set_from_geometry_ids(
            entity_type="edge",
            geometry_ids=[edge_id],
            preferred_name="Fixed Edge",
        )
        assert set_id is not None
        idx = window._load_bc_target_set_combo.findData(set_id)
        assert idx >= 0
        window._load_bc_target_set_combo.setCurrentIndex(idx)

        window._apply_bc_fixed_preset()

        assert window._scene_project.boundary_definitions
        definition = next(iter(window._scene_project.boundary_definitions.values()))
        assert definition.target_entity_type == "edge"
        assert definition.direction == "xy"
        assert abs(float(definition.value)) <= 1e-12
    finally:
        window.close()


def test_mainwindow_geometry_point_set_tool_collects_multiple_points() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        window._toggle_pick_geometry_point_set_multi()
        assert window._pending_pick_context == "geometry_point_set_multi"

        window._on_canvas_sketch_face_picked(0.0, 0.0)
        window._on_canvas_sketch_face_picked(2.0, 0.0)

        target_set = window._current_load_bc_target_set()
        assert target_set is not None
        assert target_set.entity_type == "point"
        assert target_set.binding_mode == "geometry"
        assert len(target_set.entity_ids) >= 2
    finally:
        window.close()


def test_mainwindow_geometry_split_edge_tool_creates_edge_point() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        window._toggle_pick_split_edge_point()
        assert window._pending_pick_context == "geometry_split_edge_point"
        window._on_canvas_sketch_face_picked(1.0, 0.0)

        edge_points = [
            point
            for point in window._scene_project.geometry_points.values()
            if point.source == "manual" and point.role == "edge_point"
        ]
        assert edge_points
        assert any(point.owner_edge_id for point in edge_points)
    finally:
        window.close()


def test_mainwindow_geometry_point_set_tool_reuses_same_set_id() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        window._toggle_pick_geometry_point_set_multi()
        window._on_canvas_sketch_face_picked(0.0, 0.0)
        first_set = window._current_load_bc_target_set()
        assert first_set is not None
        first_set_id = first_set.id

        window._on_canvas_sketch_face_picked(2.0, 0.0)
        second_set = window._current_load_bc_target_set()
        assert second_set is not None
        assert second_set.id == first_set_id
        assert len(window._scene_project.geometry_sets) == 1
    finally:
        window.close()


def test_mainwindow_finish_geometry_pick_exits_pending_context() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        window._toggle_pick_geometry_point_set_multi()
        assert window._pending_pick_context == "geometry_point_set_multi"

        window._finish_load_bc_geometry_picking()
        assert window._pending_pick_context is None
        assert window._mesh_canvas.current_edit_tool() == "select"
    finally:
        window.close()


def test_mainwindow_cleanup_unused_manual_geometry_points() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        created_id = window._create_geometry_edge_point_at(1.0, 0.0)
        assert created_id is not None
        assert created_id in window._scene_project.geometry_points

        window._cleanup_unused_manual_geometry_points()
        assert created_id not in window._scene_project.geometry_points
    finally:
        window.close()


def test_mainwindow_loadbc_overlay_hides_auto_sketch_vertices_by_default() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        window._sync_load_bc_geometry_overlay()
        assert window._mesh_canvas._load_bc_geometry_points == []
        assert window._mesh_canvas._load_bc_geometry_edges
    finally:
        window.close()


def test_mainwindow_loadbc_overlay_hides_circle_arc_sampling_vertices() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._create_rect_model_from_form()
        window._on_canvas_sketch_primitive_drawn("circle", 0.0, 0.0, 2.0, 2.0)

        mode_idx = window._load_bc_target_mode_combo.findData("geometry")
        assert mode_idx >= 0
        window._load_bc_target_mode_combo.setCurrentIndex(mode_idx)

        window._sync_load_bc_geometry_overlay()

        assert window._mesh_canvas._load_bc_geometry_points == []
        assert len(window._mesh_canvas._load_bc_geometry_edges) >= 24
    finally:
        window.close()


def test_mainwindow_sketch_region_pick_works_without_solver_model() -> None:
    _get_app()
    window = MainWindow(show_startup_dialog=False, language="en_US")
    try:
        window._part_sketch_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
        window._refresh_part_sketch_table()
        window._model = None

        window._start_pick_face_region()
        assert window._pending_pick_context == "material_face_sketch"

        window._pending_pick_context = None
        window._start_pick_component_face_region()
        assert window._pending_pick_context == "component_face_sketch"
    finally:
        window.close()
