
from __future__ import annotations

import copy
import importlib.util
import sys
from math import atan2, ceil, cos, hypot, radians, sin, sqrt
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QInputDialog,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QSizePolicy,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fem_ai_solver.fem.model import Mesh, Model
from fem_ai_solver.fem.model import BoundaryCondition, Element, Load, Material, Node
from fem_ai_solver.fem.results import StaticSolveResult
from fem_ai_solver.fem.services import solve_linear_static
from fem_ai_solver.mesh.importers import (
    GmshT3ImportResult,
    load_t3_mesh_from_gmsh_msh_with_physical_groups,
    load_t3_mesh_from_json,
)
from fem_ai_solver.mesh.workflow import (
    MeshControl,
    MeshGeometryEdge,
    MeshGeometryPoint,
    MeshGenerationRequest,
    MeshQualityReport,
    MeshRegion,
    MeshSeed,
    evaluate_t3_mesh_quality,
    mesher_backend,
)
from fem_ai_solver.mesh.generation import (
    BoundaryLoop,
    ConstrainedDelaunayInput,
    RegionMarker,
    generate_constrained_delaunay_t3,
    triangulate_polygon_ear_clipping,
)
from fem_ai_solver.geotech import (
    GeotechTemplateInput,
    PointOfInterest,
    build_template_input_from_text_description,
    load_geotech_layers_from_csv,
    run_geotech_template_workflow,
)
from fem_ai_solver.preprocessing.demo_model import (
    apply_left_clamp_right_nodal_force,
    build_stage1_demo_model,
    model_from_t3_mesh,
)
from fem_ai_solver.preprocessing.scene_model import (
    BoundaryDefinition,
    ComponentDef,
    GeometryEdgeDef,
    GeometryPointDef,
    GeometrySetDef,
    LoopDef,
    LoadDefinition,
    RegionDef,
    SceneMeshState,
    SceneProject,
    SceneSelection,
    polygon_area as scene_polygon_area,
    polygon_centroid,
)
from fem_ai_solver.preprocessing.model_services import (
    PhysicalGroupMappingEntry,
    TemplateApplicationReport,
    apply_default_templates_from_physical_groups,
    build_physical_group_mapping_preview,
    collect_pre_solve_warnings,
    summarize_mapping_entries,
    summarize_model_state,
    validate_model,
)
from fem_ai_solver.ui.i18n import available_languages, load_language, normalize_language, save_language, ui_text
from fem_ai_solver.ui.mesh_canvas import MeshCanvas
from fem_ai_solver.ui.result_table_utils import build_element_rows, build_node_rows, export_rows_to_csv, filter_row_indices
from fem_ai_solver.ui.startup_dialog import StartupDialog
from fem_ai_solver.ui.theme import PALETTE, apply_engineering_light_theme, mapping_row_color, status_badge_style


class MainWindow(QMainWindow):
    def __init__(self, *, show_startup_dialog: bool = True, language: str | None = None) -> None:
        super().__init__()

        self._settings = QSettings()
        self._language = normalize_language(language or load_language(self._settings))

        self._model: Model | None = None
        self._result: StaticSolveResult | None = None
        self._last_gmsh_import: GmshT3ImportResult | None = None

        self._node_rows: list[tuple[int, float, float]] = []
        self._element_rows: list[tuple[int, float, float, float, float, float, float]] = []
        self._node_visible_indices: list[int] = []
        self._element_visible_indices: list[int] = []
        self._mapping_entries: list[PhysicalGroupMappingEntry] = []

        self._mapping_applied = False
        self._results_stale = False
        self._runtime_state = "ready"
        self._is_retranslating = False
        self._geotech_point_report_text = ""
        self._geotech_last_output_dir: str | None = None
        self._geotech_last_template: GeotechTemplateInput | None = None
        self._selected_node_ids: set[int] = set()
        self._selected_element_ids: set[int] = set()
        self._part_sketch_points: list[tuple[float, float]] = []
        self._part_sketch_history: list[dict[str, object]] = []
        self._part_sketch_curve_hint: dict[str, object] | None = None
        self._part_sketch_undo_stack: list[dict[str, object]] = []
        self._part_sketch_redo_stack: list[dict[str, object]] = []
        self._selected_part_segment: tuple[int, int] | None = None
        self._selected_part_angle: tuple[int, int, int] | None = None
        self._capture_sketch_from_canvas = False
        self._last_generated_report = ""
        self._selected_face_region_element_ids: set[int] = set()
        self._selected_face_region_sketch_keys: set[str] = set()
        self._selected_geometry_point_ids: set[str] = set()
        self._selected_geometry_edge_ids: set[str] = set()
        self._last_geometry_pick_set_id: str | None = None
        self._material_geometry_region_rules: dict[str, dict[str, object]] = {}
        self._material_name_to_id: dict[str, int] = {}
        self._material_id_to_name: dict[int, str] = {}
        self._is_syncing_material_widgets = False
        self._active_sketch_width = 30.0
        self._active_sketch_height = 20.0
        self._active_sketch_grid_step = 1.5
        self._updating_part_shape_controls = False
        self._part_shape_ratio = self._active_sketch_width / max(self._active_sketch_height, 1e-6)
        self._suppress_part_selection_changed = False
        self._part_sketch_geometry_signature = ""
        self._active_workflow_step: str | None = None
        self._pending_pick_context: str | None = None
        self._geometry_pick_point_cache: list[tuple[str, float, float]] = []
        self._geometry_pick_edge_cache: list[tuple[str, float, float, float, float, float, float, float, float]] = []
        self._sketch_face_pick_cache: list[dict[str, object]] = []
        self._scene_project = SceneProject()
        self._scene_project.ensure_default_component()
        self._scene_selection = SceneSelection()
        self._scene_mesh_state = SceneMeshState()
        self._resolved_scene_loads_by_def_id: dict[str, list[Load]] = {}
        self._resolved_scene_bcs_by_def_id: dict[str, list[BoundaryCondition]] = {}
        self._pending_load_definition_name: str | None = None
        self._pending_boundary_definition_name: str | None = None
        self._pending_load_definition_step: str | None = None
        self._pending_boundary_definition_step: str | None = None
        self._pending_boundary_definition_type: str | None = None
        self._results_scope_type = "all"
        self._results_scope_target_id = ""
        self._last_mesh_quality_report: MeshQualityReport | None = None
        self._mesh_bad_elements_visible = False

        self.resize(1680, 980)
        self._build_ui()
        self._mesh_canvas.set_language(self._language)
        apply_engineering_light_theme(self)
        self._retranslate_ui()
        self._set_runtime_state("ready", self._tr("label.no_result", "No solved results yet."))

        if show_startup_dialog:
            QTimer.singleShot(0, self._show_startup_dialog)
        else:
            self._create_empty_model()

    def _tr(self, key: str, fallback: str) -> str:
        return ui_text(key, self._language, fallback)

    def _ui(self, zh_text: str, en_text: str) -> str:
        return zh_text if self._language == "zh_CN" else en_text

    def _state_text(self, state: str) -> str:
        token = state.lower().strip()
        return self._tr(f"state.{token}", token.capitalize())

    def _resolve_brand_icon(self) -> QIcon:
        base_dirs = [
            Path.cwd(),
            Path(__file__).resolve().parents[3],
        ]
        candidate_paths: list[Path] = []
        for base in base_dirs:
            for folder in ("assets", "docs/assets"):
                for name in ("rockfem_logo", "rockfem_icon", "logo", "app_icon"):
                    for ext in ("png", "jpg", "jpeg", "ico"):
                        candidate_paths.append(base / folder / f"{name}.{ext}")
        for path in candidate_paths:
            if path.exists():
                icon = QIcon(str(path))
                if not icon.isNull():
                    return icon
        return self.style().standardIcon(QStyle.SP_ComputerIcon)

    def _build_header_rows(self) -> None:
        header = QWidget(self)
        header.setObjectName("HeaderWidget")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(6)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(10)

        self._brand_icon_label = QLabel()
        self._brand_icon_label.setObjectName("BrandIcon")
        self._brand_icon_label.setFixedSize(34, 34)
        brand_icon = self._resolve_brand_icon()
        self.setWindowIcon(brand_icon)
        self._brand_icon_label.setPixmap(brand_icon.pixmap(24, 24))
        brand_row.addWidget(self._brand_icon_label)

        brand_text_col = QVBoxLayout()
        brand_text_col.setContentsMargins(0, 0, 0, 0)
        brand_text_col.setSpacing(0)
        self._brand_title_label = QLabel("RockFEM")
        self._brand_title_label.setObjectName("BrandTitle")
        self._brand_subtitle_label = QLabel()
        self._brand_subtitle_label.setObjectName("SubtleLabel")
        brand_text_col.addWidget(self._brand_title_label)
        brand_text_col.addWidget(self._brand_subtitle_label)
        brand_row.addLayout(brand_text_col)

        brand_row.addStretch(1)

        self._brand_runtime_badge = QLabel()
        self._brand_runtime_badge.setObjectName("StatusBadge")
        brand_row.addWidget(self._brand_runtime_badge)

        header_layout.addLayout(brand_row)

        self._text_menu_bar = QMenuBar()
        self._text_menu_bar.setNativeMenuBar(False)
        header_layout.addWidget(self._text_menu_bar)
        self._build_text_menu_actions()

        self._workflow_strip = QWidget()
        self._workflow_strip.setObjectName("WorkflowStrip")
        workflow_layout = QHBoxLayout(self._workflow_strip)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.setSpacing(6)

        self._workflow_buttons: dict[str, QPushButton] = {}
        for step in (
            "model",
            "part",
            "material",
            "assembly",
            "load_bc",
            "mesh",
            "job",
            "visualization",
            "report",
        ):
            button = QPushButton()
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, key=step: self._activate_workflow(key))
            self._workflow_buttons[step] = button
            workflow_layout.addWidget(button)
        workflow_layout.addStretch(1)
        header_layout.addWidget(self._workflow_strip)

        self._quick_tool_strip = QWidget()
        self._quick_tool_strip.setObjectName("QuickToolStrip")
        quick_layout = QHBoxLayout(self._quick_tool_strip)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_layout.setSpacing(6)

        self._btn_quick_undo = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_ArrowBack),
            self._undo_last_sketch_point,
        )
        quick_layout.addWidget(self._btn_quick_undo)
        self._btn_quick_redo = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_ArrowForward),
            self._redo_last_sketch_point,
        )
        quick_layout.addWidget(self._btn_quick_redo)
        self._btn_quick_zoom_in = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            self._zoom_canvas_in,
        )
        quick_layout.addWidget(self._btn_quick_zoom_in)
        self._btn_quick_zoom_out = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogListView),
            self._zoom_canvas_out,
        )
        quick_layout.addWidget(self._btn_quick_zoom_out)
        self._btn_quick_reset_view = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_BrowserReload),
            self._reset_canvas_view,
        )
        quick_layout.addWidget(self._btn_quick_reset_view)
        self._btn_quick_cancel = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogCancelButton),
            self._cancel_interactive_modes,
        )
        quick_layout.addWidget(self._btn_quick_cancel)
        quick_layout.addStretch(1)
        header_layout.addWidget(self._quick_tool_strip)

        self._part_sketch_strip = QWidget()
        self._part_sketch_strip.setObjectName("PartToolStrip")
        part_strip_layout = QHBoxLayout(self._part_sketch_strip)
        part_strip_layout.setContentsMargins(0, 0, 0, 0)
        part_strip_layout.setSpacing(6)

        self._btn_part_tool_select = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogResetButton),
            lambda: self._set_canvas_tool("select"),
        )
        part_strip_layout.addWidget(self._btn_part_tool_select)
        self._btn_part_tool_add_point = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogNewFolder),
            self._enable_part_canvas_add_point,
        )
        part_strip_layout.addWidget(self._btn_part_tool_add_point)
        self._btn_part_tool_measure_len = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogYesButton),
            self._activate_measure_tool,
        )
        part_strip_layout.addWidget(self._btn_part_tool_measure_len)
        self._btn_part_tool_measure_ang = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogHelpButton),
            self._activate_angle_measure_tool,
        )
        part_strip_layout.addWidget(self._btn_part_tool_measure_ang)
        self._btn_part_tool_rect = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_TitleBarNormalButton),
            self._activate_draw_rect_tool,
        )
        part_strip_layout.addWidget(self._btn_part_tool_rect)
        self._btn_part_tool_circle = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogYesButton),
            self._activate_draw_circle_tool,
        )
        part_strip_layout.addWidget(self._btn_part_tool_circle)
        self._btn_part_tool_ellipse = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_BrowserReload),
            self._activate_draw_ellipse_tool,
        )
        part_strip_layout.addWidget(self._btn_part_tool_ellipse)
        self._btn_part_tool_close = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogApplyButton),
            self._close_part_sketch_loop,
        )
        part_strip_layout.addWidget(self._btn_part_tool_close)
        self._btn_part_tool_clear_selected = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogDiscardButton),
            self._clear_selected_part_geometry,
        )
        part_strip_layout.addWidget(self._btn_part_tool_clear_selected)
        self._btn_part_tool_clear_all = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_TrashIcon),
            self._clear_all_part_geometry,
        )
        part_strip_layout.addWidget(self._btn_part_tool_clear_all)

        self._part_shape_w_label = QLabel("W")
        self._part_shape_w_label.setObjectName("SubtleLabel")
        part_strip_layout.addWidget(self._part_shape_w_label)
        self._part_shape_w = QDoubleSpinBox()
        self._part_shape_w.setDecimals(4)
        self._part_shape_w.setRange(1e-4, 1e9)
        self._part_shape_w.setValue(max(self._active_sketch_width * 0.35, 1e-3))
        self._part_shape_w.setMinimumWidth(95)
        self._part_shape_w.valueChanged.connect(self._on_part_shape_w_changed)
        part_strip_layout.addWidget(self._part_shape_w)
        self._part_shape_h_label = QLabel("H")
        self._part_shape_h_label.setObjectName("SubtleLabel")
        part_strip_layout.addWidget(self._part_shape_h_label)
        self._part_shape_h = QDoubleSpinBox()
        self._part_shape_h.setDecimals(4)
        self._part_shape_h.setRange(1e-4, 1e9)
        self._part_shape_h.setValue(max(self._active_sketch_height * 0.35, 1e-3))
        self._part_shape_h.setMinimumWidth(95)
        self._part_shape_h.valueChanged.connect(self._on_part_shape_h_changed)
        part_strip_layout.addWidget(self._part_shape_h)
        self._part_shape_lock_ratio = QCheckBox()
        self._part_shape_lock_ratio.toggled.connect(self._on_part_shape_lock_toggled)
        part_strip_layout.addWidget(self._part_shape_lock_ratio)

        self._part_quick_mode = QComboBox()
        self._part_quick_mode.addItem("", "length")
        self._part_quick_mode.addItem("", "angle")
        self._part_quick_mode.currentIndexChanged.connect(self._on_part_quick_mode_changed)
        self._part_quick_mode.setMinimumWidth(100)
        part_strip_layout.addWidget(self._part_quick_mode)
        self._part_quick_value = QDoubleSpinBox()
        self._part_quick_value.setDecimals(6)
        self._part_quick_value.setRange(1e-6, 1e9)
        self._part_quick_value.setValue(1.0)
        self._part_quick_value.setMinimumWidth(120)
        part_strip_layout.addWidget(self._part_quick_value)
        self._btn_part_quick_apply = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogOkButton),
            self._apply_quick_dimension_value,
        )
        part_strip_layout.addWidget(self._btn_part_quick_apply)
        part_strip_layout.addStretch(1)
        self._part_sketch_strip.setVisible(False)
        header_layout.addWidget(self._part_sketch_strip)

        self._material_tool_strip = QWidget()
        self._material_tool_strip.setObjectName("MaterialToolStrip")
        material_strip_layout = QHBoxLayout(self._material_tool_strip)
        material_strip_layout.setContentsMargins(0, 0, 0, 0)
        material_strip_layout.setSpacing(6)

        self._material_toolbar_label = QLabel()
        self._material_toolbar_label.setObjectName("SubtleLabel")
        material_strip_layout.addWidget(self._material_toolbar_label)

        self._material_toolbar_combo = QComboBox()
        self._material_toolbar_combo.setMinimumWidth(240)
        self._material_toolbar_combo.currentIndexChanged.connect(self._on_material_toolbar_combo_changed)
        material_strip_layout.addWidget(self._material_toolbar_combo)

        self._material_toolbar_template_combo = QComboBox()
        self._material_toolbar_template_combo.setMinimumWidth(150)
        self._material_toolbar_template_combo.currentIndexChanged.connect(self._on_material_template_changed)
        material_strip_layout.addWidget(self._material_toolbar_template_combo)

        self._btn_material_tool_apply_template = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            self._apply_selected_material_template,
        )
        material_strip_layout.addWidget(self._btn_material_tool_apply_template)

        self._btn_material_tool_create = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogApplyButton),
            self._create_isotropic_material,
        )
        material_strip_layout.addWidget(self._btn_material_tool_create)
        self._btn_material_tool_rename = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogInfoView),
            self._rename_selected_material,
        )
        material_strip_layout.addWidget(self._btn_material_tool_rename)
        self._btn_material_tool_duplicate = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogNewFolder),
            self._duplicate_selected_material,
        )
        material_strip_layout.addWidget(self._btn_material_tool_duplicate)
        self._btn_material_tool_pick_face = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogContentsView),
            self._start_pick_face_region,
        )
        material_strip_layout.addWidget(self._btn_material_tool_pick_face)
        self._btn_material_tool_assign = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogOkButton),
            self._assign_material_to_region,
        )
        material_strip_layout.addWidget(self._btn_material_tool_assign)
        self._btn_material_tool_delete = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_TrashIcon),
            self._delete_selected_material,
        )
        material_strip_layout.addWidget(self._btn_material_tool_delete)

        self._material_toolbar_summary = QLabel()
        self._material_toolbar_summary.setObjectName("SubtleLabel")
        material_strip_layout.addWidget(self._material_toolbar_summary)
        material_strip_layout.addStretch(1)
        self._material_tool_strip.setVisible(False)
        header_layout.addWidget(self._material_tool_strip)

        self._load_bc_command_strip = QWidget()
        self._load_bc_command_strip.setObjectName("LoadBcCommandStrip")
        load_bc_strip_layout = QHBoxLayout(self._load_bc_command_strip)
        load_bc_strip_layout.setContentsMargins(0, 0, 0, 0)
        load_bc_strip_layout.setSpacing(6)

        self._load_bc_command_label = QLabel()
        self._load_bc_command_label.setObjectName("SubtleLabel")
        load_bc_strip_layout.addWidget(self._load_bc_command_label)

        self._btn_loadbc_tool_create_load = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogNewFolder),
            self._open_create_load_dialog,
        )
        load_bc_strip_layout.addWidget(self._btn_loadbc_tool_create_load)
        self._btn_loadbc_tool_load_manager = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            self._open_load_manager_dialog,
        )
        load_bc_strip_layout.addWidget(self._btn_loadbc_tool_load_manager)
        self._btn_loadbc_tool_create_bc = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogApplyButton),
            self._open_create_boundary_dialog,
        )
        load_bc_strip_layout.addWidget(self._btn_loadbc_tool_create_bc)
        self._btn_loadbc_tool_bc_manager = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogContentsView),
            self._open_boundary_manager_dialog,
        )
        load_bc_strip_layout.addWidget(self._btn_loadbc_tool_bc_manager)
        self._load_bc_command_separator = QLabel("|")
        self._load_bc_command_separator.setObjectName("SubtleLabel")
        load_bc_strip_layout.addWidget(self._load_bc_command_separator)
        self._btn_loadbc_tool_pick_point = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_ArrowRight),
            self._start_pick_geometry_point_target,
        )
        load_bc_strip_layout.addWidget(self._btn_loadbc_tool_pick_point)
        self._btn_loadbc_tool_pick_edge = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_ArrowDown),
            self._start_pick_geometry_edge_target,
        )
        load_bc_strip_layout.addWidget(self._btn_loadbc_tool_pick_edge)
        self._btn_loadbc_tool_finish = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogOkButton),
            self._finish_load_bc_geometry_picking,
        )
        load_bc_strip_layout.addWidget(self._btn_loadbc_tool_finish)
        self._btn_loadbc_tool_cancel = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogCancelButton),
            self._cancel_interactive_modes,
        )
        load_bc_strip_layout.addWidget(self._btn_loadbc_tool_cancel)

        for button in (
            self._btn_loadbc_tool_create_load,
            self._btn_loadbc_tool_load_manager,
            self._btn_loadbc_tool_create_bc,
            self._btn_loadbc_tool_bc_manager,
            self._btn_loadbc_tool_pick_point,
            self._btn_loadbc_tool_pick_edge,
            self._btn_loadbc_tool_finish,
            self._btn_loadbc_tool_cancel,
        ):
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        load_bc_strip_layout.addStretch(1)
        self._load_bc_command_strip.setVisible(False)
        header_layout.addWidget(self._load_bc_command_strip)

        self.setMenuWidget(header)

    def _make_quick_tool_button(self, icon: QIcon, handler) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(icon)
        button.setAutoRaise(False)
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.clicked.connect(handler)
        return button

    def _build_text_menu_actions(self) -> None:
        self._menu_file = self._text_menu_bar.addMenu("")
        self._menu_part = self._text_menu_bar.addMenu("")
        self._menu_material = self._text_menu_bar.addMenu("")
        self._menu_tool = self._text_menu_bar.addMenu("")
        self._menu_view = self._text_menu_bar.addMenu("")
        self._menu_visualization = self._text_menu_bar.addMenu("")
        self._menu_help = self._text_menu_bar.addMenu("")
        self._menu_language = self._text_menu_bar.addMenu("")

        self._action_file_new = QAction(self)
        self._action_file_new.triggered.connect(self._create_empty_model)
        self._menu_file.addAction(self._action_file_new)

        self._action_file_import = QAction(self)
        self._action_file_import.triggered.connect(self._import_model_file)
        self._menu_file.addAction(self._action_file_import)

        self._action_file_export = QAction(self)
        self._action_file_export.triggered.connect(self._export_current_results)
        self._menu_file.addAction(self._action_file_export)

        self._action_file_exit = QAction(self)
        self._action_file_exit.triggered.connect(self.close)
        self._menu_file.addAction(self._action_file_exit)

        self._action_part_jump = QAction(self)
        self._action_part_jump.triggered.connect(lambda: self._activate_workflow("part"))
        self._menu_part.addAction(self._action_part_jump)
        self._action_part_build = QAction(self)
        self._action_part_build.triggered.connect(self._build_part_from_sketch)
        self._menu_part.addAction(self._action_part_build)

        self._action_material_new = QAction(self)
        self._action_material_new.triggered.connect(self._create_isotropic_material)
        self._menu_material.addAction(self._action_material_new)
        self._action_material_assign = QAction(self)
        self._action_material_assign.triggered.connect(self._assign_material_to_region)
        self._menu_material.addAction(self._action_material_assign)

        self._action_tool_export_canvas = QAction(self)
        self._action_tool_export_canvas.triggered.connect(self._export_canvas_image)
        self._menu_tool.addAction(self._action_tool_export_canvas)

        self._action_tool_layout_transform = QAction(self)
        self._action_tool_layout_transform.triggered.connect(self._open_layout_transform_tool)
        self._menu_tool.addAction(self._action_tool_layout_transform)

        self._action_tool_toggle_log = QAction(self)
        self._action_tool_toggle_log.triggered.connect(self._toggle_log_panel)
        self._menu_tool.addAction(self._action_tool_toggle_log)

        self._action_view_workspace = QAction(self)
        self._action_view_workspace.triggered.connect(lambda: self._side_tabs.setCurrentWidget(self._view_tab))
        self._menu_view.addAction(self._action_view_workspace)

        self._action_view_results = QAction(self)
        self._action_view_results.triggered.connect(lambda: self._side_tabs.setCurrentWidget(self._results_tab))
        self._menu_view.addAction(self._action_view_results)

        self._action_visualization_disp = QAction(self)
        self._action_visualization_disp.triggered.connect(lambda: self._set_contour_variable("u_mag"))
        self._menu_visualization.addAction(self._action_visualization_disp)

        self._action_visualization_mises = QAction(self)
        self._action_visualization_mises.triggered.connect(lambda: self._set_contour_variable("mise"))
        self._menu_visualization.addAction(self._action_visualization_mises)

        self._action_visualization_principal = QAction(self)
        self._action_visualization_principal.triggered.connect(lambda: self._set_contour_variable("s1"))
        self._menu_visualization.addAction(self._action_visualization_principal)

        self._action_help_about = QAction(self)
        self._action_help_about.triggered.connect(self._show_about_dialog)
        self._menu_help.addAction(self._action_help_about)

        self._action_help_workflow = QAction(self)
        self._action_help_workflow.triggered.connect(self._show_workflow_help)
        self._menu_help.addAction(self._action_help_workflow)

        self._action_lang_zh = QAction(self)
        self._action_lang_zh.setCheckable(True)
        self._action_lang_zh.triggered.connect(lambda: self._set_language_from_menu("zh_CN"))
        self._menu_language.addAction(self._action_lang_zh)
        self._action_lang_en = QAction(self)
        self._action_lang_en.setCheckable(True)
        self._action_lang_en.triggered.connect(lambda: self._set_language_from_menu("en_US"))
        self._menu_language.addAction(self._action_lang_en)

    def _show_model_group(self, group: QWidget | None = None) -> None:
        self._side_tabs.setCurrentWidget(self._model_tab)
        if group is None:
            return
        self._model_scroll.ensureWidgetVisible(group, 0, 24)

    def _show_visualization_workspace(self) -> None:
        self._side_tabs.setCurrentWidget(self._view_tab)

    def _open_layout_transform_tool(self) -> None:
        self._activate_workflow("assembly")
        self._show_model_group(self._rock_assembly_group)

    def _activate_workflow(self, step: str) -> None:
        previous = self._active_workflow_step
        if previous in {"part", "material"} and step != previous:
            self._cancel_interactive_modes()
        self._active_workflow_step = step
        for key, button in self._workflow_buttons.items():
            button.setChecked(key == step)
        self._set_part_toolbar_visible(step == "part")
        self._set_material_toolbar_visible(step == "material")
        self._set_load_bc_toolbar_visible(step == "load_bc")

        if step == "model":
            self._show_model_group(self._rock_model_group)
        elif step == "part":
            self._show_model_group(self._rock_part_group)
        elif step == "material":
            self._show_model_group(self._rock_material_group)
        elif step == "assembly":
            self._show_model_group(self._rock_assembly_group)
        elif step == "load_bc":
            self._show_model_group(self._rock_load_bc_group)
        elif step == "mesh":
            self._show_model_group(self._rock_mesh_group)
        elif step == "job":
            self._show_model_group(self._rock_job_group)
        elif step == "visualization":
            self._show_visualization_workspace()
        elif step == "report":
            self._show_model_group(self._rock_report_group)

    def _set_part_toolbar_visible(self, visible: bool) -> None:
        if hasattr(self, "_part_sketch_strip"):
            self._part_sketch_strip.setVisible(bool(visible))

    def _set_material_toolbar_visible(self, visible: bool) -> None:
        if hasattr(self, "_material_tool_strip"):
            self._material_tool_strip.setVisible(bool(visible))

    def _set_load_bc_toolbar_visible(self, visible: bool) -> None:
        if hasattr(self, "_load_bc_command_strip"):
            self._load_bc_command_strip.setVisible(bool(visible))

    def _enable_part_canvas_add_point(self) -> None:
        self._activate_workflow("part")
        self._part_capture_from_canvas.setChecked(True)
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("add_node")

    def _set_language_from_menu(self, language: str) -> None:
        normalized = normalize_language(language)
        if normalized == self._language:
            return
        self._language = normalized
        save_language(self._language, self._settings)
        self._mesh_canvas.set_language(self._language)
        self._retranslate_ui()

    def _show_about_dialog(self) -> None:
        QMessageBox.information(
            self,
            "RockFEM",
            "RockFEM 2D - AI Assisted FEM Workbench\n"
            "Current stage focuses on 2D preprocessing, solve and visualization.",
        )

    def _show_workflow_help(self) -> None:
        QMessageBox.information(
            self,
            "RockFEM Workflow",
            "Recommended sequence:\n"
            "Model -> Part -> Material -> Assembly -> Load/BC -> Mesh -> Job -> Visualization -> Report",
        )

    def _build_ui(self) -> None:
        self._build_header_rows()
        self._build_toolbar()

        center = QWidget()
        center_layout = QHBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        self._mesh_canvas = MeshCanvas()
        self._mesh_canvas.node_picked.connect(self._on_canvas_node_picked)
        self._mesh_canvas.element_picked.connect(self._on_canvas_element_picked)
        self._mesh_canvas.node_created.connect(self._on_canvas_node_created)
        self._mesh_canvas.node_moved.connect(self._on_canvas_node_moved)
        self._mesh_canvas.element_created.connect(self._on_canvas_element_created)
        self._mesh_canvas.measurement_updated.connect(self._on_measurement_updated)
        self._mesh_canvas.measurement_angle_updated.connect(self._on_measurement_angle_updated)
        self._mesh_canvas.sketch_segment_picked.connect(self._on_canvas_sketch_segment_picked)
        self._mesh_canvas.sketch_angle_picked.connect(self._on_canvas_sketch_angle_picked)
        self._mesh_canvas.sketch_history_segment_picked.connect(self._on_canvas_sketch_history_segment_picked)
        self._mesh_canvas.sketch_face_picked.connect(self._on_canvas_sketch_face_picked)
        self._mesh_canvas.sketch_primitive_drawn.connect(self._on_canvas_sketch_primitive_drawn)
        self._mesh_canvas.cancel_requested.connect(self._cancel_interactive_modes)

        self._command_history = QPlainTextEdit()
        self._command_history.setReadOnly(True)
        self._command_history.setMaximumHeight(180)
        self._command_history.setObjectName("CommandHistory")
        self._command_history.setPlainText(self._tr("log.history.ready", "RockFEM command history is ready."))

        left_split = QSplitter(Qt.Vertical)
        left_split.addWidget(self._mesh_canvas)
        left_split.addWidget(self._command_history)
        left_split.setSizes([820, 160])

        self._side_tabs = QTabWidget()
        self._side_tabs.setObjectName("SidePanel")
        self._side_tabs.setMinimumWidth(500)

        self._model_tab = self._build_model_tab()
        self._mapping_tab = self._build_mapping_tab()
        self._view_tab = self._build_view_tab()
        self._results_tab = self._build_results_tab()
        self._log_tab = self._build_log_tab()

        self._side_tabs.addTab(self._model_tab, "")
        self._side_tabs.addTab(self._mapping_tab, "")
        self._side_tabs.addTab(self._view_tab, "")
        self._side_tabs.addTab(self._results_tab, "")
        self._side_tabs.addTab(self._log_tab, "")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left_split)
        split.addWidget(self._side_tabs)
        split.setSizes([1120, 520])

        center_layout.addWidget(split)
        self.setCentralWidget(center)
        self._build_log_dock()
        self._setup_keyboard_shortcuts()
        self._toolbar.hide()
        self._activate_workflow("model")

    def _setup_keyboard_shortcuts(self) -> None:
        self._shortcut_escape = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._shortcut_escape.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_escape.activated.connect(self._cancel_interactive_modes)
        self._shortcut_undo = QShortcut(QKeySequence.Undo, self)
        self._shortcut_undo.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_undo.activated.connect(self._undo_last_sketch_point)
        self._shortcut_redo = QShortcut(QKeySequence.Redo, self)
        self._shortcut_redo.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_redo.activated.connect(self._redo_last_sketch_point)
        self._shortcut_zoom_in = QShortcut(QKeySequence.ZoomIn, self)
        self._shortcut_zoom_in.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_zoom_in.activated.connect(self._zoom_canvas_in)
        self._shortcut_zoom_out = QShortcut(QKeySequence.ZoomOut, self)
        self._shortcut_zoom_out.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_zoom_out.activated.connect(self._zoom_canvas_out)

    def _cancel_interactive_modes(self) -> None:
        """Cancel temporary canvas/edit modes (sketch capture, pick context, previews)."""
        had_active_state = bool(
            self._capture_sketch_from_canvas
            or self._pending_pick_context is not None
            or self._mesh_canvas.current_edit_tool() != "select"
        )

        self._mesh_canvas.cancel_temporary_interactions()
        self._selected_part_segment = None
        self._selected_part_angle = None

        if getattr(self, "_part_capture_from_canvas", None) is not None and self._part_capture_from_canvas.isChecked():
            self._part_capture_from_canvas.setChecked(False)
        else:
            self._capture_sketch_from_canvas = False

        self._pending_pick_context = None
        if hasattr(self, "_material_face_hint"):
            self._material_face_hint.setText(self._tr("rock.material.pick_face.hint", "未选择区域"))
        if hasattr(self, "_component_face_hint"):
            self._component_face_hint.setText(self._tr("rock.assembly.component.pick.hint", "未选择区域"))
        self._selected_geometry_edge_ids.clear()
        if hasattr(self, "_mesh_selection_hint"):
            self._update_mesh_selection_hint()
        if hasattr(self, "_load_bc_set_hint") and self._current_load_bc_target_mode() == "geometry":
            self._load_bc_set_hint.setText(
                self._ui("已退出几何拾取模式。", "Geometry pick mode exited.")
            )

        self._set_canvas_tool("select")
        self._sync_pickable_sketch_faces_to_canvas()
        self._sync_load_bc_geometry_overlay()

        if had_active_state:
            self._log(self._ui("已取消当前画布绘制/拾取操作（Esc）。", "Cancelled current canvas draw/pick operation (Esc)."))

    def _zoom_canvas_in(self) -> None:
        if hasattr(self, "_mesh_canvas"):
            self._mesh_canvas.zoom_in()

    def _zoom_canvas_out(self) -> None:
        if hasattr(self, "_mesh_canvas"):
            self._mesh_canvas.zoom_out()

    def _reset_canvas_view(self) -> None:
        if hasattr(self, "_mesh_canvas"):
            self._mesh_canvas.reset_view()

    def _build_toolbar(self) -> None:
        self._toolbar = QToolBar()
        self._toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self._toolbar)

        self._btn_toolbar_demo = self._make_toolbar_button("", self.style().standardIcon(QStyle.SP_DialogResetButton), self._load_demo_model)
        self._toolbar.addWidget(self._btn_toolbar_demo)

        self._btn_toolbar_geotech = QToolButton()
        self._btn_toolbar_geotech.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self._btn_toolbar_geotech.setPopupMode(QToolButton.MenuButtonPopup)
        self._toolbar_geotech_menu = QMenu(self)
        self._action_geotech_quick = QAction(self)
        self._action_geotech_quick.triggered.connect(self._run_geotech_quick_template)
        self._toolbar_geotech_menu.addAction(self._action_geotech_quick)
        self._action_geotech_from_text = QAction(self)
        self._action_geotech_from_text.triggered.connect(self._run_geotech_from_text_dialog)
        self._toolbar_geotech_menu.addAction(self._action_geotech_from_text)
        self._action_geotech_from_csv = QAction(self)
        self._action_geotech_from_csv.triggered.connect(self._run_geotech_from_csv_dialog)
        self._toolbar_geotech_menu.addAction(self._action_geotech_from_csv)
        self._btn_toolbar_geotech.setMenu(self._toolbar_geotech_menu)
        self._btn_toolbar_geotech.clicked.connect(self._run_geotech_quick_template)
        self._toolbar.addWidget(self._btn_toolbar_geotech)

        self._btn_toolbar_import = QToolButton()
        self._btn_toolbar_import.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self._btn_toolbar_import.setPopupMode(QToolButton.MenuButtonPopup)
        self._toolbar_import_menu = QMenu(self)
        self._action_import_json = QAction(self)
        self._action_import_json.triggered.connect(self._import_json_mesh)
        self._toolbar_import_menu.addAction(self._action_import_json)
        self._action_import_msh = QAction(self)
        self._action_import_msh.triggered.connect(self._import_gmsh_mesh)
        self._toolbar_import_menu.addAction(self._action_import_msh)
        self._btn_toolbar_import.setMenu(self._toolbar_import_menu)
        self._btn_toolbar_import.clicked.connect(self._import_model_file)
        self._toolbar.addWidget(self._btn_toolbar_import)

        self._btn_toolbar_solve = self._make_toolbar_button("", self.style().standardIcon(QStyle.SP_MediaPlay), self._submit_job, primary=True)
        self._toolbar.addWidget(self._btn_toolbar_solve)

        self._btn_toolbar_export = self._make_toolbar_button("", self.style().standardIcon(QStyle.SP_DialogSaveButton), self._export_current_results)
        self._toolbar.addWidget(self._btn_toolbar_export)

        self._toolbar.addSeparator()

        self._backend_label = QLabel()
        self._backend_label.setObjectName("SubtleLabel")
        self._toolbar.addWidget(self._backend_label)
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["python", "cpp"])
        self._backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self._toolbar.addWidget(self._backend_combo)

        self._toolbar.addSeparator()

        self._language_label = QLabel()
        self._language_label.setObjectName("SubtleLabel")
        self._toolbar.addWidget(self._language_label)
        self._language_combo = QComboBox()
        self._language_combo.currentIndexChanged.connect(self._on_language_changed)
        self._toolbar.addWidget(self._language_combo)

        self._toolbar.addSeparator()
        self._toolbar_runtime_badge = QLabel()
        self._toolbar_runtime_badge.setObjectName("StatusBadge")
        self._toolbar.addWidget(self._toolbar_runtime_badge)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._toolbar.addWidget(spacer)

        self._btn_toggle_log = self._make_toolbar_button("", self.style().standardIcon(QStyle.SP_FileDialogInfoView), self._toggle_log_panel)
        self._btn_toggle_log.setCheckable(True)
        self._toolbar.addWidget(self._btn_toggle_log)

    def _make_toolbar_button(self, text: str, icon, handler, *, primary: bool = False) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setIcon(icon)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.clicked.connect(handler)
        if primary:
            button.setObjectName("PrimaryToolButton")
        return button

    def _build_model_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        scroll = QScrollArea()
        self._model_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status_group = QGroupBox()
        status_layout = QFormLayout(self._status_group)
        self._status_node_count = QLabel("0")
        self._status_element_count = QLabel("0")
        self._status_material_count = QLabel("0")
        self._status_bc_count = QLabel("0")
        self._status_load_count = QLabel("0")
        self._status_backend = QLabel("python")

        self._lbl_nodes = QLabel()
        self._lbl_elements = QLabel()
        self._lbl_materials = QLabel()
        self._lbl_bc = QLabel()
        self._lbl_loads = QLabel()
        self._lbl_backend = QLabel()

        status_layout.addRow(self._lbl_nodes, self._status_node_count)
        status_layout.addRow(self._lbl_elements, self._status_element_count)
        status_layout.addRow(self._lbl_materials, self._status_material_count)
        status_layout.addRow(self._lbl_bc, self._status_bc_count)
        status_layout.addRow(self._lbl_loads, self._status_load_count)
        status_layout.addRow(self._lbl_backend, self._status_backend)

        self._runtime_group = QGroupBox()
        runtime_layout = QVBoxLayout(self._runtime_group)
        self._runtime_badge = QLabel()
        self._runtime_badge.setObjectName("StatusBadge")
        self._runtime_detail = QLabel()
        self._runtime_detail.setWordWrap(True)
        self._runtime_detail.setObjectName("SubtleLabel")
        self._pre_solve_warning = QLabel()
        self._pre_solve_warning.setWordWrap(True)
        self._pre_solve_warning.setObjectName("SubtleLabel")
        runtime_layout.addWidget(self._runtime_badge)
        runtime_layout.addWidget(self._runtime_detail)
        runtime_layout.addWidget(self._pre_solve_warning)

        self._rock_model_group = self._build_rock_model_group()
        self._rock_part_group = self._build_rock_part_group()
        self._rock_material_group = self._build_rock_material_group()
        self._rock_assembly_group = self._build_rock_assembly_group()
        self._rock_load_bc_group = self._build_rock_load_bc_group()
        self._rock_mesh_group = self._build_rock_mesh_group()
        self._rock_job_group = self._build_rock_job_group()
        self._rock_report_group = self._build_rock_report_group()

        self._geotech_group = QGroupBox()
        geotech_layout = QVBoxLayout(self._geotech_group)
        self._geotech_info = QLabel()
        self._geotech_info.setWordWrap(True)
        self._geotech_info.setObjectName("SubtleLabel")
        geotech_layout.addWidget(self._geotech_info)

        geotech_param_form = QFormLayout()
        self._geotech_mesh_size_label = QLabel()
        self._geotech_mesh_size_spin = QDoubleSpinBox()
        self._geotech_mesh_size_spin.setDecimals(3)
        self._geotech_mesh_size_spin.setRange(0.05, 1000.0)
        self._geotech_mesh_size_spin.setSingleStep(0.1)
        self._geotech_mesh_size_spin.setValue(1.0)
        geotech_param_form.addRow(self._geotech_mesh_size_label, self._geotech_mesh_size_spin)

        self._geotech_top_load_label = QLabel()
        self._geotech_top_load_spin = QDoubleSpinBox()
        self._geotech_top_load_spin.setDecimals(3)
        self._geotech_top_load_spin.setRange(-1_000_000.0, 1_000_000.0)
        self._geotech_top_load_spin.setSingleStep(5.0)
        self._geotech_top_load_spin.setValue(30.0)
        geotech_param_form.addRow(self._geotech_top_load_label, self._geotech_top_load_spin)
        geotech_layout.addLayout(geotech_param_form)

        self._geotech_fix_bottom_ux = QCheckBox()
        self._geotech_fix_bottom_ux.setChecked(True)
        geotech_layout.addWidget(self._geotech_fix_bottom_ux)

        self._geotech_fix_bottom_uy = QCheckBox()
        self._geotech_fix_bottom_uy.setChecked(True)
        geotech_layout.addWidget(self._geotech_fix_bottom_uy)

        self._geotech_fix_lateral_ux = QCheckBox()
        self._geotech_fix_lateral_ux.setChecked(True)
        geotech_layout.addWidget(self._geotech_fix_lateral_ux)

        self._geotech_export_artifacts = QCheckBox()
        self._geotech_export_artifacts.setChecked(True)
        geotech_layout.addWidget(self._geotech_export_artifacts)

        self._btn_geotech_quick = QPushButton()
        self._btn_geotech_quick.clicked.connect(self._run_geotech_quick_template)
        geotech_layout.addWidget(self._btn_geotech_quick)

        self._btn_geotech_text = QPushButton()
        self._btn_geotech_text.clicked.connect(self._run_geotech_from_text_dialog)
        geotech_layout.addWidget(self._btn_geotech_text)

        self._btn_geotech_csv = QPushButton()
        self._btn_geotech_csv.clicked.connect(self._run_geotech_from_csv_dialog)
        geotech_layout.addWidget(self._btn_geotech_csv)

        geotech_aux_row = QHBoxLayout()
        self._btn_geotech_rerun = QPushButton()
        self._btn_geotech_rerun.clicked.connect(self._rerun_last_geotech_template)
        geotech_aux_row.addWidget(self._btn_geotech_rerun)

        self._btn_geotech_open_output = QPushButton()
        self._btn_geotech_open_output.clicked.connect(self._open_geotech_output_folder)
        geotech_aux_row.addWidget(self._btn_geotech_open_output)
        geotech_layout.addLayout(geotech_aux_row)

        self._geotech_point_summary = QLabel()
        self._geotech_point_summary.setWordWrap(True)
        self._geotech_point_summary.setObjectName("SubtleLabel")
        geotech_layout.addWidget(self._geotech_point_summary)

        self._editor_group = QGroupBox()
        editor_layout = QVBoxLayout(self._editor_group)
        self._editor_info = QLabel()
        self._editor_info.setWordWrap(True)
        self._editor_info.setObjectName("SubtleLabel")
        editor_layout.addWidget(self._editor_info)

        canvas_edit_row = QHBoxLayout()
        self._editor_canvas_edit_enable = QCheckBox()
        self._editor_canvas_edit_enable.toggled.connect(self._on_canvas_edit_enabled_changed)
        canvas_edit_row.addWidget(self._editor_canvas_edit_enable)
        self._editor_canvas_tool_label = QLabel()
        canvas_edit_row.addWidget(self._editor_canvas_tool_label)
        self._editor_canvas_tool_combo = QComboBox()
        self._editor_canvas_tool_combo.currentIndexChanged.connect(self._on_canvas_edit_tool_changed)
        canvas_edit_row.addWidget(self._editor_canvas_tool_combo)
        self._editor_canvas_material_label = QLabel()
        canvas_edit_row.addWidget(self._editor_canvas_material_label)
        self._editor_canvas_material_spin = QSpinBox()
        self._editor_canvas_material_spin.setRange(1, 1_000_000)
        self._editor_canvas_material_spin.setValue(1)
        canvas_edit_row.addWidget(self._editor_canvas_material_spin)
        editor_layout.addLayout(canvas_edit_row)

        editor_actions_row = QHBoxLayout()
        self._btn_editor_load = QPushButton()
        self._btn_editor_load.clicked.connect(self._sync_editor_tables_from_model)
        editor_actions_row.addWidget(self._btn_editor_load)
        self._btn_editor_validate = QPushButton()
        self._btn_editor_validate.clicked.connect(self._validate_editor_draft)
        editor_actions_row.addWidget(self._btn_editor_validate)
        self._btn_editor_apply = QPushButton()
        self._btn_editor_apply.clicked.connect(self._apply_editor_draft_to_model)
        editor_actions_row.addWidget(self._btn_editor_apply)
        editor_layout.addLayout(editor_actions_row)

        self._editor_material_table = QTableWidget(0, 4)
        self._editor_material_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        editor_layout.addWidget(self._editor_material_table)
        mat_actions = QHBoxLayout()
        self._btn_editor_add_material = QPushButton()
        self._btn_editor_add_material.clicked.connect(
            lambda: self._append_editor_row(self._editor_material_table, ["", "", "", "false"])
        )
        mat_actions.addWidget(self._btn_editor_add_material)
        self._btn_editor_del_material = QPushButton()
        self._btn_editor_del_material.clicked.connect(
            lambda: self._delete_selected_editor_rows(self._editor_material_table)
        )
        mat_actions.addWidget(self._btn_editor_del_material)
        mat_actions.addStretch(1)
        editor_layout.addLayout(mat_actions)

        self._editor_node_table = QTableWidget(0, 3)
        self._editor_node_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        editor_layout.addWidget(self._editor_node_table)
        node_actions = QHBoxLayout()
        self._btn_editor_add_node = QPushButton()
        self._btn_editor_add_node.clicked.connect(
            lambda: self._append_editor_row(self._editor_node_table, ["", "", ""])
        )
        node_actions.addWidget(self._btn_editor_add_node)
        self._btn_editor_del_node = QPushButton()
        self._btn_editor_del_node.clicked.connect(
            lambda: self._delete_selected_editor_rows(self._editor_node_table)
        )
        node_actions.addWidget(self._btn_editor_del_node)
        node_actions.addStretch(1)
        editor_layout.addLayout(node_actions)

        self._editor_element_table = QTableWidget(0, 5)
        self._editor_element_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        editor_layout.addWidget(self._editor_element_table)
        elem_actions = QHBoxLayout()
        self._btn_editor_add_element = QPushButton()
        self._btn_editor_add_element.clicked.connect(
            lambda: self._append_editor_row(self._editor_element_table, ["", "", "", "", "1"])
        )
        elem_actions.addWidget(self._btn_editor_add_element)
        self._btn_editor_del_element = QPushButton()
        self._btn_editor_del_element.clicked.connect(
            lambda: self._delete_selected_editor_rows(self._editor_element_table)
        )
        elem_actions.addWidget(self._btn_editor_del_element)
        elem_actions.addStretch(1)
        editor_layout.addLayout(elem_actions)

        self._explorer_group = QGroupBox()
        explorer_layout = QVBoxLayout(self._explorer_group)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("Model")
        explorer_layout.addWidget(self._tree)

        layout.addWidget(self._status_group)
        layout.addWidget(self._runtime_group)
        layout.addWidget(self._rock_model_group)
        layout.addWidget(self._rock_part_group)
        layout.addWidget(self._rock_material_group)
        layout.addWidget(self._rock_assembly_group)
        layout.addWidget(self._rock_load_bc_group)
        layout.addWidget(self._rock_mesh_group)
        layout.addWidget(self._rock_job_group)
        layout.addWidget(self._rock_report_group)
        layout.addWidget(self._geotech_group)
        layout.addWidget(self._editor_group)
        layout.addWidget(self._explorer_group)
        layout.addStretch(1)

        scroll.setWidget(scroll_content)
        tab_layout.addWidget(scroll)
        return tab

    def _build_rock_model_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        self._rock_model_intro = QLabel()
        self._rock_model_intro.setWordWrap(True)
        self._rock_model_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._rock_model_intro)

        action_row = QHBoxLayout()
        self._btn_rock_ai_image = QPushButton()
        self._btn_rock_ai_image.clicked.connect(self._import_ai_model_image)
        action_row.addWidget(self._btn_rock_ai_image)
        self._btn_rock_ai_context = QPushButton()
        self._btn_rock_ai_context.clicked.connect(self._run_ai_context_model)
        action_row.addWidget(self._btn_rock_ai_context)
        layout.addLayout(action_row)

        self._rock_ai_image_path = QLineEdit()
        self._rock_ai_image_path.setReadOnly(True)
        layout.addWidget(self._rock_ai_image_path)

        self._rock_ai_context_text = QPlainTextEdit()
        self._rock_ai_context_text.setPlaceholderText("输入工程语境，例如：边坡分层、基础埋深、荷载等。")
        self._rock_ai_context_text.setFixedHeight(80)
        layout.addWidget(self._rock_ai_context_text)

        form = QFormLayout()
        self._rock_model_width_label = QLabel("Width")
        self._rock_rect_width = QDoubleSpinBox()
        self._rock_rect_width.setRange(0.1, 100000.0)
        self._rock_rect_width.setValue(30.0)
        form.addRow(self._rock_model_width_label, self._rock_rect_width)

        self._rock_model_height_label = QLabel("Height")
        self._rock_rect_height = QDoubleSpinBox()
        self._rock_rect_height.setRange(0.1, 100000.0)
        self._rock_rect_height.setValue(20.0)
        form.addRow(self._rock_model_height_label, self._rock_rect_height)

        self._rock_model_seed_label = QLabel("Sketch Grid Step")
        self._rock_rect_seed = QDoubleSpinBox()
        self._rock_rect_seed.setDecimals(3)
        self._rock_rect_seed.setRange(0.05, 5000.0)
        self._rock_rect_seed.setValue(1.5)
        form.addRow(self._rock_model_seed_label, self._rock_rect_seed)
        layout.addLayout(form)

        self._btn_rock_create_2d = QPushButton()
        self._btn_rock_create_2d.clicked.connect(self._create_rect_model_from_form)
        layout.addWidget(self._btn_rock_create_2d)

        tools_row = QHBoxLayout()
        self._btn_model_measure = QPushButton()
        self._btn_model_measure.clicked.connect(self._activate_measure_tool)
        tools_row.addWidget(self._btn_model_measure)
        self._btn_model_reset_view = QPushButton()
        self._btn_model_reset_view.clicked.connect(self._reset_canvas_view)
        tools_row.addWidget(self._btn_model_reset_view)
        layout.addLayout(tools_row)
        self._model_measure_result = QLabel()
        self._model_measure_result.setObjectName("SubtleLabel")
        layout.addWidget(self._model_measure_result)
        return group

    def _build_rock_part_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        self._rock_part_intro = QLabel()
        self._rock_part_intro.setWordWrap(True)
        self._rock_part_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._rock_part_intro)

        point_row = QHBoxLayout()
        self._part_point_x = QDoubleSpinBox()
        self._part_point_x.setRange(-100000.0, 100000.0)
        self._part_point_x.setDecimals(4)
        self._part_point_x.setValue(0.0)
        point_row.addWidget(self._part_point_x)
        self._part_point_y = QDoubleSpinBox()
        self._part_point_y.setRange(-100000.0, 100000.0)
        self._part_point_y.setDecimals(4)
        self._part_point_y.setValue(0.0)
        point_row.addWidget(self._part_point_y)
        self._btn_part_add_point = QPushButton()
        self._btn_part_add_point.clicked.connect(self._add_part_sketch_point)
        point_row.addWidget(self._btn_part_add_point)
        layout.addLayout(point_row)

        self._part_capture_from_canvas = QCheckBox()
        self._part_capture_from_canvas.toggled.connect(self._on_capture_sketch_toggled)
        layout.addWidget(self._part_capture_from_canvas)
        self._part_snap_to_grid = QCheckBox()
        self._part_snap_to_grid.setChecked(True)
        layout.addWidget(self._part_snap_to_grid)

        self._part_sketch_table = QTableWidget(0, 3)
        self._part_sketch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._part_sketch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._part_sketch_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._part_sketch_table.itemSelectionChanged.connect(self._on_part_sketch_selection_changed)
        layout.addWidget(self._part_sketch_table)

        button_row = QHBoxLayout()
        self._btn_part_clear_selected = QPushButton()
        self._btn_part_clear_selected.clicked.connect(self._clear_selected_part_geometry)
        button_row.addWidget(self._btn_part_clear_selected)
        self._btn_part_clear = QPushButton()
        self._btn_part_clear.clicked.connect(self._clear_part_sketch)
        button_row.addWidget(self._btn_part_clear)
        self._btn_part_close = QPushButton()
        self._btn_part_close.clicked.connect(self._close_part_sketch_loop)
        button_row.addWidget(self._btn_part_close)
        self._btn_part_build = QPushButton()
        self._btn_part_build.clicked.connect(self._build_part_from_sketch)
        button_row.addWidget(self._btn_part_build)
        layout.addLayout(button_row)

        self._part_toolbox_group = QGroupBox()
        part_toolbox_layout = QVBoxLayout(self._part_toolbox_group)

        measure_row = QHBoxLayout()
        self._btn_part_measure = QPushButton()
        self._btn_part_measure.clicked.connect(self._activate_measure_tool)
        measure_row.addWidget(self._btn_part_measure)
        self._btn_part_measure_fill_length = QPushButton()
        self._btn_part_measure_fill_length.clicked.connect(self._fill_length_from_measurement)
        measure_row.addWidget(self._btn_part_measure_fill_length)
        measure_row.addStretch(1)
        part_toolbox_layout.addLayout(measure_row)

        length_row = QHBoxLayout()
        self._part_length_target_label = QLabel()
        length_row.addWidget(self._part_length_target_label)
        self._part_length_target = QDoubleSpinBox()
        self._part_length_target.setDecimals(6)
        self._part_length_target.setRange(1e-6, 1e9)
        self._part_length_target.setValue(1.0)
        length_row.addWidget(self._part_length_target)
        self._btn_part_apply_length = QPushButton()
        self._btn_part_apply_length.clicked.connect(self._apply_part_length_constraint)
        length_row.addWidget(self._btn_part_apply_length)
        part_toolbox_layout.addLayout(length_row)

        angle_row = QHBoxLayout()
        self._part_angle_target_label = QLabel()
        angle_row.addWidget(self._part_angle_target_label)
        self._part_angle_target = QDoubleSpinBox()
        self._part_angle_target.setDecimals(3)
        self._part_angle_target.setRange(1.0, 179.0)
        self._part_angle_target.setValue(90.0)
        angle_row.addWidget(self._part_angle_target)
        self._btn_part_apply_angle = QPushButton()
        self._btn_part_apply_angle.clicked.connect(self._apply_part_angle_constraint)
        angle_row.addWidget(self._btn_part_apply_angle)
        part_toolbox_layout.addLayout(angle_row)

        constraint_row1 = QHBoxLayout()
        self._btn_part_constraint_horizontal = QPushButton()
        self._btn_part_constraint_horizontal.clicked.connect(self._apply_segment_horizontal_constraint)
        constraint_row1.addWidget(self._btn_part_constraint_horizontal)
        self._btn_part_constraint_vertical = QPushButton()
        self._btn_part_constraint_vertical.clicked.connect(self._apply_segment_vertical_constraint)
        constraint_row1.addWidget(self._btn_part_constraint_vertical)
        self._btn_part_constraint_collinear = QPushButton()
        self._btn_part_constraint_collinear.clicked.connect(self._apply_points_collinear_constraint)
        constraint_row1.addWidget(self._btn_part_constraint_collinear)
        part_toolbox_layout.addLayout(constraint_row1)

        constraint_row2 = QHBoxLayout()
        self._btn_part_constraint_parallel = QPushButton()
        self._btn_part_constraint_parallel.clicked.connect(self._apply_segments_parallel_constraint)
        constraint_row2.addWidget(self._btn_part_constraint_parallel)
        self._btn_part_constraint_perpendicular = QPushButton()
        self._btn_part_constraint_perpendicular.clicked.connect(self._apply_segments_perpendicular_constraint)
        constraint_row2.addWidget(self._btn_part_constraint_perpendicular)
        constraint_row2.addStretch(1)
        part_toolbox_layout.addLayout(constraint_row2)

        self._part_dimension_status = QLabel()
        self._part_dimension_status.setWordWrap(True)
        self._part_dimension_status.setObjectName("SubtleLabel")
        part_toolbox_layout.addWidget(self._part_dimension_status)

        layout.addWidget(self._part_toolbox_group)
        return group

    def _build_rock_material_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        self._rock_material_intro = QLabel()
        self._rock_material_intro.setWordWrap(True)
        self._rock_material_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._rock_material_intro)

        self._material_action_toolbar = QWidget()
        self._material_action_toolbar.setObjectName("MaterialActionToolbar")
        action_toolbar_layout = QHBoxLayout(self._material_action_toolbar)
        action_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        action_toolbar_layout.setSpacing(6)
        self._material_action_template_label = QLabel()
        self._material_action_template_label.setObjectName("SubtleLabel")
        action_toolbar_layout.addWidget(self._material_action_template_label)
        self._material_action_template_combo = QComboBox()
        self._material_action_template_combo.setMinimumWidth(180)
        self._material_action_template_combo.currentIndexChanged.connect(self._on_material_template_changed)
        action_toolbar_layout.addWidget(self._material_action_template_combo)
        self._btn_material_action_apply_template = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            self._apply_selected_material_template,
        )
        action_toolbar_layout.addWidget(self._btn_material_action_apply_template)
        self._btn_material_action_create = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogApplyButton),
            self._create_isotropic_material,
        )
        action_toolbar_layout.addWidget(self._btn_material_action_create)
        self._btn_material_action_rename = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogInfoView),
            self._rename_selected_material,
        )
        action_toolbar_layout.addWidget(self._btn_material_action_rename)
        self._btn_material_action_duplicate = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogNewFolder),
            self._duplicate_selected_material,
        )
        action_toolbar_layout.addWidget(self._btn_material_action_duplicate)
        self._btn_material_action_delete = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_TrashIcon),
            self._delete_selected_material,
        )
        action_toolbar_layout.addWidget(self._btn_material_action_delete)
        self._btn_material_action_pick_face = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_FileDialogContentsView),
            self._start_pick_face_region,
        )
        action_toolbar_layout.addWidget(self._btn_material_action_pick_face)
        self._btn_material_action_assign = self._make_quick_tool_button(
            self.style().standardIcon(QStyle.SP_DialogOkButton),
            self._assign_material_to_region,
        )
        action_toolbar_layout.addWidget(self._btn_material_action_assign)
        action_toolbar_layout.addStretch(1)
        layout.addWidget(self._material_action_toolbar)

        form = QFormLayout()
        self._material_name_label = QLabel("Material Name")
        self._material_name_edit = QLineEdit("Soil-1")
        form.addRow(self._material_name_label, self._material_name_edit)

        self._material_e_label = QLabel("Elastic E")
        self._material_e_spin = QDoubleSpinBox()
        self._material_e_spin.setRange(1.0, 1e15)
        self._material_e_spin.setDecimals(3)
        self._material_e_spin.setValue(2.0e7)
        form.addRow(self._material_e_label, self._material_e_spin)

        self._material_nu_label = QLabel("Poisson nu")
        self._material_nu_spin = QDoubleSpinBox()
        self._material_nu_spin.setRange(-0.99, 0.49)
        self._material_nu_spin.setDecimals(4)
        self._material_nu_spin.setSingleStep(0.01)
        self._material_nu_spin.setValue(0.30)
        form.addRow(self._material_nu_label, self._material_nu_spin)

        self._material_plane_stress_label = QLabel("Plane Stress")
        self._material_plane_stress = QCheckBox()
        self._material_plane_stress.setChecked(True)
        form.addRow(self._material_plane_stress_label, self._material_plane_stress)
        layout.addLayout(form)

        self._material_unit_hint = QLabel()
        self._material_unit_hint.setWordWrap(True)
        self._material_unit_hint.setObjectName("SubtleLabel")
        layout.addWidget(self._material_unit_hint)

        template_row = QHBoxLayout()
        self._material_template_label = QLabel()
        template_row.addWidget(self._material_template_label)
        self._material_template_combo = QComboBox()
        self._material_template_combo.currentIndexChanged.connect(self._on_material_template_changed)
        template_row.addWidget(self._material_template_combo)
        self._btn_material_apply_template = QPushButton()
        self._btn_material_apply_template.clicked.connect(self._apply_selected_material_template)
        template_row.addWidget(self._btn_material_apply_template)
        layout.addLayout(template_row)

        self._btn_material_create = QPushButton()
        self._btn_material_create.clicked.connect(self._create_isotropic_material)
        layout.addWidget(self._btn_material_create)

        self._material_manager_group = QGroupBox()
        manager_layout = QVBoxLayout(self._material_manager_group)
        self._material_manager_summary = QLabel()
        self._material_manager_summary.setWordWrap(True)
        self._material_manager_summary.setObjectName("SubtleLabel")
        manager_layout.addWidget(self._material_manager_summary)
        self._material_manager_table = QTableWidget(0, 6)
        self._material_manager_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._material_manager_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._material_manager_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._material_manager_table.itemSelectionChanged.connect(self._on_material_manager_selection_changed)
        manager_layout.addWidget(self._material_manager_table)
        manager_actions = QHBoxLayout()
        self._btn_material_rename = QPushButton()
        self._btn_material_rename.clicked.connect(self._rename_selected_material)
        manager_actions.addWidget(self._btn_material_rename)
        self._btn_material_duplicate = QPushButton()
        self._btn_material_duplicate.clicked.connect(self._duplicate_selected_material)
        manager_actions.addWidget(self._btn_material_duplicate)
        self._btn_material_delete = QPushButton()
        self._btn_material_delete.clicked.connect(self._delete_selected_material)
        manager_actions.addWidget(self._btn_material_delete)
        manager_actions.addStretch(1)
        manager_layout.addLayout(manager_actions)

        batch_row = QHBoxLayout()
        self._material_reassign_target_label = QLabel()
        batch_row.addWidget(self._material_reassign_target_label)
        self._material_reassign_target_combo = QComboBox()
        batch_row.addWidget(self._material_reassign_target_combo)
        self._btn_material_reassign_all = QPushButton()
        self._btn_material_reassign_all.clicked.connect(self._batch_reassign_selected_material)
        batch_row.addWidget(self._btn_material_reassign_all)
        manager_layout.addLayout(batch_row)

        self._material_usage_details = QPlainTextEdit()
        self._material_usage_details.setReadOnly(True)
        self._material_usage_details.setMaximumHeight(140)
        manager_layout.addWidget(self._material_usage_details)
        layout.addWidget(self._material_manager_group)

        assign_form = QFormLayout()
        self._material_assign_id_label = QLabel("Assign Material")
        self._material_assign_combo = QComboBox()
        self._material_assign_combo.currentIndexChanged.connect(self._on_material_assign_combo_changed)
        assign_form.addRow(self._material_assign_id_label, self._material_assign_combo)
        self._material_assign_id_spin = QSpinBox()
        self._material_assign_id_spin.setRange(1, 1_000_000)
        self._material_assign_id_spin.hide()
        self._material_assign_elements_label = QLabel("Selected Region")
        self._material_assign_elements = QLineEdit()
        self._material_assign_elements.setPlaceholderText("auto")
        assign_form.addRow(self._material_assign_elements_label, self._material_assign_elements)
        layout.addLayout(assign_form)

        pick_row = QHBoxLayout()
        self._btn_material_pick_face = QPushButton()
        self._btn_material_pick_face.clicked.connect(self._start_pick_face_region)
        pick_row.addWidget(self._btn_material_pick_face)
        self._material_face_hint = QLabel()
        self._material_face_hint.setObjectName("SubtleLabel")
        pick_row.addWidget(self._material_face_hint)
        layout.addLayout(pick_row)

        self._btn_material_assign = QPushButton()
        self._btn_material_assign.clicked.connect(self._assign_material_to_region)
        layout.addWidget(self._btn_material_assign)
        return group

    def _build_rock_assembly_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        self._rock_assembly_intro = QLabel()
        self._rock_assembly_intro.setWordWrap(True)
        self._rock_assembly_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._rock_assembly_intro)

        self._component_manager_intro = QLabel()
        self._component_manager_intro.setWordWrap(True)
        self._component_manager_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._component_manager_intro)

        self._component_manager_table = QTableWidget(0, 5)
        self._component_manager_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._component_manager_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._component_manager_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._component_manager_table.itemSelectionChanged.connect(self._on_component_manager_selection_changed)
        self._component_manager_table.horizontalHeader().setStretchLastSection(True)
        self._component_manager_table.setMinimumHeight(176)
        layout.addWidget(self._component_manager_table)

        component_pick_row = QHBoxLayout()
        self._btn_component_pick_face = QPushButton()
        self._btn_component_pick_face.clicked.connect(self._start_pick_component_face_region)
        component_pick_row.addWidget(self._btn_component_pick_face)
        self._component_face_hint = QLabel()
        self._component_face_hint.setObjectName("SubtleLabel")
        component_pick_row.addWidget(self._component_face_hint)
        layout.addLayout(component_pick_row)

        component_actions = QHBoxLayout()
        self._btn_component_create_from_selected = QPushButton()
        self._btn_component_create_from_selected.clicked.connect(self._create_component_from_selected_regions)
        component_actions.addWidget(self._btn_component_create_from_selected)
        self._btn_component_rename = QPushButton()
        self._btn_component_rename.clicked.connect(self._rename_selected_component)
        component_actions.addWidget(self._btn_component_rename)
        self._btn_component_toggle_visibility = QPushButton()
        self._btn_component_toggle_visibility.clicked.connect(self._toggle_selected_component_visibility)
        component_actions.addWidget(self._btn_component_toggle_visibility)
        layout.addLayout(component_actions)

        component_actions_2 = QHBoxLayout()
        self._btn_component_isolate = QPushButton()
        self._btn_component_isolate.clicked.connect(self._isolate_selected_component)
        component_actions_2.addWidget(self._btn_component_isolate)
        self._btn_component_show_all = QPushButton()
        self._btn_component_show_all.clicked.connect(self._show_all_components)
        component_actions_2.addWidget(self._btn_component_show_all)
        component_actions_2.addStretch(1)
        layout.addLayout(component_actions_2)

        self._assembly_transform_group = QGroupBox()
        transform_layout = QVBoxLayout(self._assembly_transform_group)
        self._assembly_transform_intro = QLabel()
        self._assembly_transform_intro.setWordWrap(True)
        self._assembly_transform_intro.setObjectName("SubtleLabel")
        transform_layout.addWidget(self._assembly_transform_intro)
        form = QFormLayout()
        self._assembly_instance_label = QLabel()
        self._assembly_instance_name = QLineEdit("PART-1")
        form.addRow(self._assembly_instance_label, self._assembly_instance_name)
        self._assembly_dx_label = QLabel()
        self._assembly_dx_spin = QDoubleSpinBox()
        self._assembly_dx_spin.setRange(-100000.0, 100000.0)
        self._assembly_dx_spin.setDecimals(4)
        form.addRow(self._assembly_dx_label, self._assembly_dx_spin)
        self._assembly_dy_label = QLabel()
        self._assembly_dy_spin = QDoubleSpinBox()
        self._assembly_dy_spin.setRange(-100000.0, 100000.0)
        self._assembly_dy_spin.setDecimals(4)
        form.addRow(self._assembly_dy_label, self._assembly_dy_spin)
        transform_layout.addLayout(form)
        self._btn_apply_assembly_offset = QPushButton()
        self._btn_apply_assembly_offset.clicked.connect(self._apply_assembly_offset)
        transform_layout.addWidget(self._btn_apply_assembly_offset)
        layout.addWidget(self._assembly_transform_group)
        return group

    def _build_rock_load_bc_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        self._rock_load_bc_intro = QLabel()
        self._rock_load_bc_intro.setWordWrap(True)
        self._rock_load_bc_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._rock_load_bc_intro)

        self._load_bc_set_group = QGroupBox()
        set_layout = QVBoxLayout(self._load_bc_set_group)
        set_form = QFormLayout()
        self._load_bc_target_mode_label = QLabel()
        self._load_bc_target_mode_combo = QComboBox()
        self._load_bc_target_mode_combo.addItem("Mesh Target", "mesh")
        self._load_bc_target_mode_combo.addItem("Geometry Target", "geometry")
        self._load_bc_target_mode_combo.currentIndexChanged.connect(self._on_load_bc_target_mode_changed)
        set_form.addRow(self._load_bc_target_mode_label, self._load_bc_target_mode_combo)
        self._load_bc_target_set_label = QLabel()
        self._load_bc_target_set_combo = QComboBox()
        self._load_bc_target_set_combo.currentIndexChanged.connect(self._on_load_bc_target_set_changed)
        set_form.addRow(self._load_bc_target_set_label, self._load_bc_target_set_combo)
        set_layout.addLayout(set_form)
        set_actions = QHBoxLayout()
        self._btn_load_bc_capture_set = QPushButton()
        self._btn_load_bc_capture_set.clicked.connect(self._capture_geometry_set_from_current_target)
        set_actions.addWidget(self._btn_load_bc_capture_set)
        self._btn_load_bc_capture_region_set = QPushButton()
        self._btn_load_bc_capture_region_set.clicked.connect(self._capture_region_set_from_selected_regions)
        set_actions.addWidget(self._btn_load_bc_capture_region_set)
        self._btn_load_bc_capture_component_set = QPushButton()
        self._btn_load_bc_capture_component_set.clicked.connect(self._capture_component_set_from_selected_component)
        set_actions.addWidget(self._btn_load_bc_capture_component_set)
        self._btn_load_bc_pick_region = QPushButton()
        self._btn_load_bc_pick_region.clicked.connect(self._start_pick_load_bc_region_target)
        set_actions.addWidget(self._btn_load_bc_pick_region)
        self._btn_load_bc_pick_geo_point = QPushButton()
        self._btn_load_bc_pick_geo_point.clicked.connect(self._start_pick_geometry_point_target)
        set_actions.addWidget(self._btn_load_bc_pick_geo_point)
        self._btn_load_bc_pick_geo_edge = QPushButton()
        self._btn_load_bc_pick_geo_edge.clicked.connect(self._start_pick_geometry_edge_target)
        set_actions.addWidget(self._btn_load_bc_pick_geo_edge)
        set_actions.addStretch(1)
        set_layout.addLayout(set_actions)

        set_tools = QHBoxLayout()
        self._btn_load_bc_pick_geo_point_set = QPushButton()
        self._btn_load_bc_pick_geo_point_set.clicked.connect(self._toggle_pick_geometry_point_set_multi)
        set_tools.addWidget(self._btn_load_bc_pick_geo_point_set)
        self._btn_load_bc_split_edge_point = QPushButton()
        self._btn_load_bc_split_edge_point.clicked.connect(self._toggle_pick_split_edge_point)
        set_tools.addWidget(self._btn_load_bc_split_edge_point)
        self._btn_load_bc_create_point_by_coord = QPushButton()
        self._btn_load_bc_create_point_by_coord.clicked.connect(self._create_geometry_point_by_coordinate_dialog)
        set_tools.addWidget(self._btn_load_bc_create_point_by_coord)
        self._btn_load_bc_create_edge_fraction_point = QPushButton()
        self._btn_load_bc_create_edge_fraction_point.clicked.connect(self._create_geometry_point_on_selected_edge_fraction_dialog)
        set_tools.addWidget(self._btn_load_bc_create_edge_fraction_point)
        self._btn_load_bc_clear_point_set = QPushButton()
        self._btn_load_bc_clear_point_set.clicked.connect(self._clear_geometry_point_set_selection)
        set_tools.addWidget(self._btn_load_bc_clear_point_set)
        self._btn_load_bc_finish_pick = QPushButton()
        self._btn_load_bc_finish_pick.clicked.connect(self._finish_load_bc_geometry_picking)
        set_tools.addWidget(self._btn_load_bc_finish_pick)
        self._btn_load_bc_cleanup_points = QPushButton()
        self._btn_load_bc_cleanup_points.clicked.connect(self._cleanup_unused_manual_geometry_points)
        set_tools.addWidget(self._btn_load_bc_cleanup_points)
        set_tools.addStretch(1)
        set_layout.addLayout(set_tools)

        self._load_bc_set_hint = QLabel()
        self._load_bc_set_hint.setObjectName("SubtleLabel")
        set_layout.addWidget(self._load_bc_set_hint)
        layout.addWidget(self._load_bc_set_group)

        manager_launch_group = QGroupBox()
        manager_launch_layout = QHBoxLayout(manager_launch_group)
        self._btn_create_load_dialog = QPushButton()
        self._btn_create_load_dialog.clicked.connect(self._open_create_load_dialog)
        manager_launch_layout.addWidget(self._btn_create_load_dialog)
        self._btn_open_load_manager_dialog = QPushButton()
        self._btn_open_load_manager_dialog.clicked.connect(self._open_load_manager_dialog)
        manager_launch_layout.addWidget(self._btn_open_load_manager_dialog)
        self._btn_create_bc_dialog = QPushButton()
        self._btn_create_bc_dialog.clicked.connect(self._open_create_boundary_dialog)
        manager_launch_layout.addWidget(self._btn_create_bc_dialog)
        self._btn_open_bc_manager_dialog = QPushButton()
        self._btn_open_bc_manager_dialog.clicked.connect(self._open_boundary_manager_dialog)
        manager_launch_layout.addWidget(self._btn_open_bc_manager_dialog)
        manager_launch_layout.addStretch(1)
        layout.addWidget(manager_launch_group)
        self._load_bc_manager_launch_group = manager_launch_group

        load_tools_group = QGroupBox()
        load_tools = QHBoxLayout(load_tools_group)
        self._btn_tool_point_force = QPushButton()
        self._btn_tool_point_force.clicked.connect(self._start_pick_geometry_point_target)
        load_tools.addWidget(self._btn_tool_point_force)
        self._btn_tool_edge_load = QPushButton()
        self._btn_tool_edge_load.clicked.connect(self._start_pick_distributed_edge_target)
        load_tools.addWidget(self._btn_tool_edge_load)
        self._btn_tool_edge_pressure = QPushButton()
        self._btn_tool_edge_pressure.clicked.connect(self._prepare_edge_pressure_creation)
        load_tools.addWidget(self._btn_tool_edge_pressure)
        self._btn_tool_body_force = QPushButton()
        self._btn_tool_body_force.clicked.connect(self._prepare_region_body_force_creation)
        load_tools.addWidget(self._btn_tool_body_force)
        self._btn_tool_gravity = QPushButton()
        self._btn_tool_gravity.clicked.connect(self._prepare_gravity_creation)
        load_tools.addWidget(self._btn_tool_gravity)
        load_tools.addStretch(1)
        layout.addWidget(load_tools_group)
        self._load_bc_tool_group = load_tools_group

        concentrated = QGroupBox()
        concentrated_form = QFormLayout(concentrated)
        self._load_point_node_spin = QSpinBox()
        self._load_point_node_spin.setRange(1, 1_000_000)
        concentrated_form.addRow("Node ID", self._load_point_node_spin)
        self._load_point_vec_x = QDoubleSpinBox()
        self._load_point_vec_x.setDecimals(6)
        self._load_point_vec_x.setRange(-1e6, 1e6)
        self._load_point_vec_x.setValue(1.0)
        concentrated_form.addRow("Dir Vector X", self._load_point_vec_x)
        self._load_point_vec_y = QDoubleSpinBox()
        self._load_point_vec_y.setDecimals(6)
        self._load_point_vec_y.setRange(-1e6, 1e6)
        self._load_point_vec_y.setValue(0.0)
        concentrated_form.addRow("Dir Vector Y", self._load_point_vec_y)
        self._load_point_magnitude_spin = QDoubleSpinBox()
        self._load_point_magnitude_spin.setDecimals(6)
        self._load_point_magnitude_spin.setRange(-1e12, 1e12)
        concentrated_form.addRow("Magnitude", self._load_point_magnitude_spin)
        self._btn_pick_load_point = QPushButton()
        self._btn_pick_load_point.clicked.connect(self._start_pick_load_point)
        concentrated_form.addRow(self._btn_pick_load_point)
        self._btn_add_point_load = QPushButton()
        self._btn_add_point_load.clicked.connect(self._add_concentrated_load)
        concentrated_form.addRow(self._btn_add_point_load)
        layout.addWidget(concentrated)

        distributed = QGroupBox()
        distributed_form = QFormLayout(distributed)
        self._load_dist_type_combo = QComboBox()
        self._load_dist_type_combo.addItem("Edge Line Load (Total)", "distributed")
        self._load_dist_type_combo.addItem("Edge Pressure (Intensity)", "pressure")
        distributed_form.addRow("Load Type", self._load_dist_type_combo)
        self._load_dist_axis_combo = QComboBox()
        self._load_dist_axis_combo.addItem("x = const", "x")
        self._load_dist_axis_combo.addItem("y = const", "y")
        distributed_form.addRow("Target Axis", self._load_dist_axis_combo)
        self._load_dist_coord_spin = QDoubleSpinBox()
        self._load_dist_coord_spin.setRange(-100000.0, 100000.0)
        self._load_dist_coord_spin.setDecimals(4)
        distributed_form.addRow("Coordinate", self._load_dist_coord_spin)
        self._load_dist_tol_spin = QDoubleSpinBox()
        self._load_dist_tol_spin.setRange(1e-6, 1000.0)
        self._load_dist_tol_spin.setDecimals(6)
        self._load_dist_tol_spin.setValue(0.05)
        distributed_form.addRow("Tolerance", self._load_dist_tol_spin)
        self._load_dist_vec_x = QDoubleSpinBox()
        self._load_dist_vec_x.setDecimals(6)
        self._load_dist_vec_x.setRange(-1e6, 1e6)
        self._load_dist_vec_x.setValue(1.0)
        distributed_form.addRow("Dir Vector X", self._load_dist_vec_x)
        self._load_dist_vec_y = QDoubleSpinBox()
        self._load_dist_vec_y.setDecimals(6)
        self._load_dist_vec_y.setRange(-1e6, 1e6)
        self._load_dist_vec_y.setValue(0.0)
        distributed_form.addRow("Dir Vector Y", self._load_dist_vec_y)
        self._load_dist_direction_mode_combo = QComboBox()
        self._load_dist_direction_mode_combo.addItem("Custom Vector", "vector")
        self._load_dist_direction_mode_combo.addItem("Edge Normal", "normal")
        self._load_dist_direction_mode_combo.addItem("Reverse Edge Normal", "reverse_normal")
        distributed_form.addRow("Direction Mode", self._load_dist_direction_mode_combo)
        self._load_dist_profile_combo = QComboBox()
        self._load_dist_profile_combo.addItem("Uniform", "uniform")
        self._load_dist_profile_combo.addItem("Nonuniform (Linear)", "linear")
        distributed_form.addRow("Profile", self._load_dist_profile_combo)
        self._load_dist_total_mag_spin = QDoubleSpinBox()
        self._load_dist_total_mag_spin.setRange(-1e12, 1e12)
        self._load_dist_total_mag_spin.setDecimals(6)
        distributed_form.addRow("Total Magnitude", self._load_dist_total_mag_spin)
        self._btn_pick_dist_edge = QPushButton()
        self._btn_pick_dist_edge.clicked.connect(self._start_pick_distributed_edge_target)
        distributed_form.addRow(self._btn_pick_dist_edge)
        self._btn_add_dist_load = QPushButton()
        self._btn_add_dist_load.clicked.connect(self._add_distributed_load)
        distributed_form.addRow(self._btn_add_dist_load)
        layout.addWidget(distributed)

        region_load = QGroupBox()
        region_form = QFormLayout(region_load)
        self._region_load_vec_x = QDoubleSpinBox()
        self._region_load_vec_x.setDecimals(6)
        self._region_load_vec_x.setRange(-1e6, 1e6)
        self._region_load_vec_x.setValue(0.0)
        region_form.addRow("Body Vector X", self._region_load_vec_x)
        self._region_load_vec_y = QDoubleSpinBox()
        self._region_load_vec_y.setDecimals(6)
        self._region_load_vec_y.setRange(-1e6, 1e6)
        self._region_load_vec_y.setValue(-1.0)
        region_form.addRow("Body Vector Y", self._region_load_vec_y)
        self._region_load_intensity_spin = QDoubleSpinBox()
        self._region_load_intensity_spin.setDecimals(6)
        self._region_load_intensity_spin.setRange(-1e12, 1e12)
        region_form.addRow("Force / Area", self._region_load_intensity_spin)
        self._btn_add_body_force = QPushButton()
        self._btn_add_body_force.clicked.connect(self._add_region_body_force)
        region_form.addRow(self._btn_add_body_force)
        self._btn_add_gravity = QPushButton()
        self._btn_add_gravity.clicked.connect(self._add_region_gravity)
        region_form.addRow(self._btn_add_gravity)
        layout.addWidget(region_load)
        self._region_load_group = region_load

        bc_group = QGroupBox()
        bc_form = QFormLayout(bc_group)
        self._bc_target_combo = QComboBox()
        self._bc_target_combo.addItem("Point", "point")
        self._bc_target_combo.addItem("Edge", "edge")
        self._bc_target_combo.addItem("All Nodes (legacy)", "face")
        bc_form.addRow("Target", self._bc_target_combo)
        self._bc_point_node_spin = QSpinBox()
        self._bc_point_node_spin.setRange(1, 1_000_000)
        bc_form.addRow("Point Node", self._bc_point_node_spin)
        self._btn_pick_bc_point = QPushButton()
        self._btn_pick_bc_point.clicked.connect(self._start_pick_bc_point)
        bc_form.addRow(self._btn_pick_bc_point)
        self._btn_pick_bc_edge = QPushButton()
        self._btn_pick_bc_edge.clicked.connect(self._start_pick_bc_edge)
        bc_form.addRow(self._btn_pick_bc_edge)
        self._bc_edge_axis_combo = QComboBox()
        self._bc_edge_axis_combo.addItem("x = const", "x")
        self._bc_edge_axis_combo.addItem("y = const", "y")
        bc_form.addRow("Edge Axis", self._bc_edge_axis_combo)
        self._bc_edge_coord_spin = QDoubleSpinBox()
        self._bc_edge_coord_spin.setRange(-100000.0, 100000.0)
        self._bc_edge_coord_spin.setDecimals(4)
        bc_form.addRow("Edge Coordinate", self._bc_edge_coord_spin)
        self._bc_edge_tol_spin = QDoubleSpinBox()
        self._bc_edge_tol_spin.setRange(1e-6, 1000.0)
        self._bc_edge_tol_spin.setDecimals(6)
        self._bc_edge_tol_spin.setValue(0.05)
        bc_form.addRow("Edge Tolerance", self._bc_edge_tol_spin)
        self._bc_dir_combo = QComboBox()
        self._bc_dir_combo.addItem("X", "x")
        self._bc_dir_combo.addItem("Y", "y")
        self._bc_dir_combo.addItem("XY", "xy")
        self._bc_dir_combo.addItem("Rotation (reserved for beam/shell)", "rot")
        bc_form.addRow("DOF", self._bc_dir_combo)
        self._bc_value_spin = QDoubleSpinBox()
        self._bc_value_spin.setRange(-1e6, 1e6)
        self._bc_value_spin.setDecimals(6)
        bc_form.addRow("Value", self._bc_value_spin)
        bc_preset_row = QHBoxLayout()
        self._btn_bc_fixed = QPushButton()
        self._btn_bc_fixed.clicked.connect(self._apply_bc_fixed_preset)
        bc_preset_row.addWidget(self._btn_bc_fixed)
        self._btn_bc_fix_x = QPushButton()
        self._btn_bc_fix_x.clicked.connect(self._apply_bc_fix_x_preset)
        bc_preset_row.addWidget(self._btn_bc_fix_x)
        self._btn_bc_fix_y = QPushButton()
        self._btn_bc_fix_y.clicked.connect(self._apply_bc_fix_y_preset)
        bc_preset_row.addWidget(self._btn_bc_fix_y)
        self._btn_bc_roller_x = QPushButton()
        self._btn_bc_roller_x.clicked.connect(self._apply_bc_roller_x_preset)
        bc_preset_row.addWidget(self._btn_bc_roller_x)
        self._btn_bc_roller_y = QPushButton()
        self._btn_bc_roller_y.clicked.connect(self._apply_bc_roller_y_preset)
        bc_preset_row.addWidget(self._btn_bc_roller_y)
        bc_form.addRow(bc_preset_row)
        self._btn_apply_bc = QPushButton()
        self._btn_apply_bc.clicked.connect(self._apply_constraint)
        bc_form.addRow(self._btn_apply_bc)
        layout.addWidget(bc_group)

        manager_group = QGroupBox()
        manager_layout = QVBoxLayout(manager_group)
        self._load_manager_table = QTableWidget(0, 6)
        self._load_manager_table.setHorizontalHeaderLabels(["Name", "Type", "Target", "Mag", "Status", "Step"])
        self._load_manager_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._load_manager_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._load_manager_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        manager_layout.addWidget(self._load_manager_table)
        load_manager_row = QHBoxLayout()
        self._btn_load_manager_apply_edit = QPushButton()
        self._btn_load_manager_apply_edit.clicked.connect(self._update_selected_load_definition_from_controls)
        load_manager_row.addWidget(self._btn_load_manager_apply_edit)
        self._btn_load_manager_toggle = QPushButton()
        self._btn_load_manager_toggle.clicked.connect(self._toggle_selected_load_definition_active)
        load_manager_row.addWidget(self._btn_load_manager_toggle)
        self._btn_load_manager_locate = QPushButton()
        self._btn_load_manager_locate.clicked.connect(self._locate_selected_load_definition)
        load_manager_row.addWidget(self._btn_load_manager_locate)
        self._btn_load_manager_delete = QPushButton()
        self._btn_load_manager_delete.clicked.connect(self._delete_selected_load_definition)
        load_manager_row.addWidget(self._btn_load_manager_delete)
        load_manager_row.addStretch(1)
        manager_layout.addLayout(load_manager_row)

        self._bc_manager_table = QTableWidget(0, 6)
        self._bc_manager_table.setHorizontalHeaderLabels(["Name", "Type", "Target", "DOF", "Status", "Step"])
        self._bc_manager_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._bc_manager_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._bc_manager_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        manager_layout.addWidget(self._bc_manager_table)
        bc_manager_row = QHBoxLayout()
        self._btn_bc_manager_apply_edit = QPushButton()
        self._btn_bc_manager_apply_edit.clicked.connect(self._update_selected_boundary_definition_from_controls)
        bc_manager_row.addWidget(self._btn_bc_manager_apply_edit)
        self._btn_bc_manager_toggle = QPushButton()
        self._btn_bc_manager_toggle.clicked.connect(self._toggle_selected_boundary_definition_active)
        bc_manager_row.addWidget(self._btn_bc_manager_toggle)
        self._btn_bc_manager_locate = QPushButton()
        self._btn_bc_manager_locate.clicked.connect(self._locate_selected_boundary_definition)
        bc_manager_row.addWidget(self._btn_bc_manager_locate)
        self._btn_bc_manager_delete = QPushButton()
        self._btn_bc_manager_delete.clicked.connect(self._delete_selected_boundary_definition)
        bc_manager_row.addWidget(self._btn_bc_manager_delete)
        bc_manager_row.addStretch(1)
        manager_layout.addLayout(bc_manager_row)
        layout.addWidget(manager_group)
        self._load_bc_manager_group = manager_group
        return group

    def _build_rock_mesh_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        self._rock_mesh_intro = QLabel()
        self._rock_mesh_intro.setWordWrap(True)
        self._rock_mesh_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._rock_mesh_intro)

        form = QFormLayout()
        self._mesh_algo_combo = QComboBox()
        self._mesh_algo_combo.addItem("Structured T3", "structured")
        self._mesh_algo_combo.addItem("Delaunay T3", "delaunay")
        self._mesh_algo_combo.addItem("GitHub Ear-Clipping (Sketch)", "github_ear")
        self._mesh_algo_combo.addItem("Import Existing Mesh", "import")
        form.addRow("Algorithm", self._mesh_algo_combo)

        self._mesh_backend_label = QLabel("Backend")
        self._mesh_backend_combo = QComboBox()
        self._mesh_backend_combo.addItem("Gmsh T3", "gmsh")
        self._mesh_backend_combo.addItem("Builtin T3 (Fallback)", "builtin")
        if importlib.util.find_spec("gmsh") is None:
            self._mesh_backend_combo.setCurrentIndex(1)
        self._mesh_backend_combo.currentIndexChanged.connect(self._on_mesh_backend_changed)
        form.addRow(self._mesh_backend_label, self._mesh_backend_combo)

        self._mesh_width_spin = QDoubleSpinBox()
        self._mesh_width_spin.setRange(0.1, 100000.0)
        self._mesh_width_spin.setValue(30.0)
        form.addRow("Domain Width", self._mesh_width_spin)
        self._mesh_height_spin = QDoubleSpinBox()
        self._mesh_height_spin.setRange(0.1, 100000.0)
        self._mesh_height_spin.setValue(20.0)
        form.addRow("Domain Height", self._mesh_height_spin)

        self._mesh_global_seed_spin = QDoubleSpinBox()
        self._mesh_global_seed_spin.setRange(0.05, 5000.0)
        self._mesh_global_seed_spin.setDecimals(4)
        self._mesh_global_seed_spin.setValue(1.2)
        form.addRow("Global Seed", self._mesh_global_seed_spin)

        self._mesh_use_local_seed = QCheckBox()
        form.addRow("Enable Local Seed", self._mesh_use_local_seed)
        self._mesh_local_seed_spin = QDoubleSpinBox()
        self._mesh_local_seed_spin.setRange(0.01, 1000.0)
        self._mesh_local_seed_spin.setDecimals(4)
        self._mesh_local_seed_spin.setValue(0.6)
        form.addRow("Local Seed", self._mesh_local_seed_spin)

        self._mesh_local_xmin_spin = QDoubleSpinBox()
        self._mesh_local_xmin_spin.setRange(-100000.0, 100000.0)
        self._mesh_local_xmin_spin.setValue(8.0)
        form.addRow("Local xmin", self._mesh_local_xmin_spin)
        self._mesh_local_xmax_spin = QDoubleSpinBox()
        self._mesh_local_xmax_spin.setRange(-100000.0, 100000.0)
        self._mesh_local_xmax_spin.setValue(16.0)
        form.addRow("Local xmax", self._mesh_local_xmax_spin)
        self._mesh_local_ymin_spin = QDoubleSpinBox()
        self._mesh_local_ymin_spin.setRange(-100000.0, 100000.0)
        self._mesh_local_ymin_spin.setValue(4.0)
        form.addRow("Local ymin", self._mesh_local_ymin_spin)
        self._mesh_local_ymax_spin = QDoubleSpinBox()
        self._mesh_local_ymax_spin.setRange(-100000.0, 100000.0)
        self._mesh_local_ymax_spin.setValue(10.0)
        form.addRow("Local ymax", self._mesh_local_ymax_spin)
        layout.addLayout(form)

        self._mesh_element_type_hint = QLabel("Solvable Element Type: T3")
        self._mesh_element_type_hint.setObjectName("SubtleLabel")
        self._mesh_element_type_hint.setWordWrap(True)
        layout.addWidget(self._mesh_element_type_hint)

        seed_group = QGroupBox("Seed")
        seed_layout = QVBoxLayout(seed_group)
        seed_form = QFormLayout()
        self._mesh_region_seed_spin = QDoubleSpinBox()
        self._mesh_region_seed_spin.setRange(0.01, 5000.0)
        self._mesh_region_seed_spin.setDecimals(4)
        self._mesh_region_seed_spin.setValue(0.8)
        seed_form.addRow("Region Seed", self._mesh_region_seed_spin)
        self._mesh_edge_seed_spin = QDoubleSpinBox()
        self._mesh_edge_seed_spin.setRange(0.01, 5000.0)
        self._mesh_edge_seed_spin.setDecimals(4)
        self._mesh_edge_seed_spin.setValue(0.6)
        seed_form.addRow("Edge Seed", self._mesh_edge_seed_spin)
        seed_layout.addLayout(seed_form)
        seed_buttons = QHBoxLayout()
        self._btn_mesh_pick_region = QPushButton("Pick Region")
        self._btn_mesh_pick_region.clicked.connect(self._start_pick_mesh_region)
        seed_buttons.addWidget(self._btn_mesh_pick_region)
        self._btn_mesh_apply_region_seed = QPushButton("Apply Region Seed")
        self._btn_mesh_apply_region_seed.clicked.connect(self._apply_mesh_size_to_selected_regions)
        seed_buttons.addWidget(self._btn_mesh_apply_region_seed)
        self._btn_mesh_pick_edge = QPushButton("Pick Edge")
        self._btn_mesh_pick_edge.clicked.connect(self._start_pick_mesh_edge_seed)
        seed_buttons.addWidget(self._btn_mesh_pick_edge)
        self._btn_mesh_apply_edge_seed = QPushButton("Apply Edge Seed")
        self._btn_mesh_apply_edge_seed.clicked.connect(self._apply_mesh_size_to_selected_edges)
        seed_buttons.addWidget(self._btn_mesh_apply_edge_seed)
        seed_layout.addLayout(seed_buttons)
        layout.addWidget(seed_group)

        edit_group = QGroupBox("Edit Mesh")
        edit_layout = QVBoxLayout(edit_group)
        edit_buttons = QHBoxLayout()
        self._btn_mesh_generate_selection = QPushButton("Generate Selected")
        self._btn_mesh_generate_selection.clicked.connect(self._generate_selected_region_mesh)
        edit_buttons.addWidget(self._btn_mesh_generate_selection)
        self._btn_mesh_remesh_selection = QPushButton("Remesh Selected")
        self._btn_mesh_remesh_selection.clicked.connect(self._remesh_selected_region_mesh)
        edit_buttons.addWidget(self._btn_mesh_remesh_selection)
        self._btn_mesh_delete_selection = QPushButton("Delete Selected")
        self._btn_mesh_delete_selection.clicked.connect(self._delete_selected_mesh_elements)
        edit_buttons.addWidget(self._btn_mesh_delete_selection)
        edit_layout.addLayout(edit_buttons)
        self._mesh_selection_hint = QLabel("No local mesh target selected.")
        self._mesh_selection_hint.setWordWrap(True)
        self._mesh_selection_hint.setObjectName("SubtleLabel")
        edit_layout.addWidget(self._mesh_selection_hint)
        layout.addWidget(edit_group)

        quality_group = QGroupBox("Quality")
        quality_layout = QVBoxLayout(quality_group)
        quality_buttons = QHBoxLayout()
        self._btn_mesh_quality_check = QPushButton("Check Quality")
        self._btn_mesh_quality_check.clicked.connect(self._check_mesh_quality)
        quality_buttons.addWidget(self._btn_mesh_quality_check)
        self._btn_mesh_toggle_bad = QPushButton("Highlight Bad")
        self._btn_mesh_toggle_bad.clicked.connect(self._toggle_bad_mesh_highlight)
        quality_buttons.addWidget(self._btn_mesh_toggle_bad)
        quality_layout.addLayout(quality_buttons)
        self._mesh_quality_summary = QLabel("No mesh quality report yet.")
        self._mesh_quality_summary.setWordWrap(True)
        self._mesh_quality_summary.setObjectName("SubtleLabel")
        quality_layout.addWidget(self._mesh_quality_summary)
        layout.addWidget(quality_group)

        mesh_row = QHBoxLayout()
        self._btn_generate_mesh = QPushButton()
        self._btn_generate_mesh.clicked.connect(self._generate_mesh_from_controls)
        mesh_row.addWidget(self._btn_generate_mesh)
        self._btn_import_mesh = QPushButton()
        self._btn_import_mesh.clicked.connect(self._import_model_file)
        mesh_row.addWidget(self._btn_import_mesh)
        layout.addLayout(mesh_row)
        return group

    def _build_rock_job_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        self._rock_job_intro = QLabel()
        self._rock_job_intro.setWordWrap(True)
        self._rock_job_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._rock_job_intro)

        form = QFormLayout()
        self._job_name_edit = QLineEdit("Job-1")
        form.addRow("Job Name", self._job_name_edit)
        layout.addLayout(form)

        job_row = QHBoxLayout()
        self._btn_submit_job = QPushButton()
        self._btn_submit_job.clicked.connect(self._submit_job)
        job_row.addWidget(self._btn_submit_job)
        self._btn_job_export_canvas = QPushButton()
        self._btn_job_export_canvas.clicked.connect(self._export_canvas_image)
        job_row.addWidget(self._btn_job_export_canvas)
        layout.addLayout(job_row)

        self._job_status_label = QLabel()
        self._job_status_label.setObjectName("SubtleLabel")
        layout.addWidget(self._job_status_label)
        return group

    def _build_rock_report_group(self) -> QGroupBox:
        group = QGroupBox()
        layout = QVBoxLayout(group)

        self._rock_report_intro = QLabel()
        self._rock_report_intro.setWordWrap(True)
        self._rock_report_intro.setObjectName("SubtleLabel")
        layout.addWidget(self._rock_report_intro)

        form = QFormLayout()
        self._report_standard_combo = QComboBox()
        self._report_standard_combo.addItems(
            [
                "GB 50007 地基基础",
                "JGJ 94 建筑桩基",
                "SL 191 水工混凝土",
            ]
        )
        form.addRow("Standard", self._report_standard_combo)
        layout.addLayout(form)

        self._report_context_edit = QPlainTextEdit()
        self._report_context_edit.setFixedHeight(80)
        self._report_context_edit.setPlaceholderText("补充工程语境：场地条件、设计要求、风险点。")
        layout.addWidget(self._report_context_edit)

        self._report_metric_label = QLabel()
        self._report_metric_label.setWordWrap(True)
        self._report_metric_label.setObjectName("SubtleLabel")
        layout.addWidget(self._report_metric_label)

        actions = QHBoxLayout()
        self._btn_generate_report = QPushButton()
        self._btn_generate_report.clicked.connect(self._generate_engineering_report)
        actions.addWidget(self._btn_generate_report)
        self._btn_export_report = QPushButton()
        self._btn_export_report.clicked.connect(self._export_report_text)
        actions.addWidget(self._btn_export_report)
        layout.addLayout(actions)

        self._report_output = QPlainTextEdit()
        self._report_output.setReadOnly(True)
        self._report_output.setMinimumHeight(160)
        layout.addWidget(self._report_output)
        return group

    def _build_mapping_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self._mapping_summary_group = QGroupBox()
        summary_layout = QVBoxLayout(self._mapping_summary_group)
        counts_row = QHBoxLayout()

        self._mapping_total_badge = QLabel("total 0")
        self._mapping_mapped_badge = QLabel("mapped 0")
        self._mapping_warning_badge = QLabel("warning 0")
        self._mapping_unrec_badge = QLabel("unrecognized 0")
        for badge in (self._mapping_total_badge, self._mapping_mapped_badge, self._mapping_warning_badge, self._mapping_unrec_badge):
            badge.setObjectName("StatusBadge")
            counts_row.addWidget(badge)
        counts_row.addStretch(1)
        summary_layout.addLayout(counts_row)

        self._mapping_summary = QLabel()
        self._mapping_summary.setObjectName("SubtleLabel")
        self._mapping_summary.setWordWrap(True)
        summary_layout.addWidget(self._mapping_summary)

        self._mapping_risk_text = QLabel()
        self._mapping_risk_text.setWordWrap(True)
        self._mapping_risk_text.setObjectName("SubtleLabel")
        summary_layout.addWidget(self._mapping_risk_text)

        self._btn_apply_mapping = QPushButton()
        self._btn_apply_mapping.clicked.connect(self._apply_mapping_from_last_import)
        summary_layout.addWidget(self._btn_apply_mapping)

        self._mapping_table_group = QGroupBox()
        table_layout = QVBoxLayout(self._mapping_table_group)
        self._mapping_table = QTableWidget(0, 7)
        self._mapping_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._mapping_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._mapping_table.setSortingEnabled(True)
        table_layout.addWidget(self._mapping_table)

        layout.addWidget(self._mapping_summary_group)
        layout.addWidget(self._mapping_table_group)
        return tab

    def _build_view_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self._vis_metrics_label = QLabel()
        self._vis_metrics_label.setObjectName("SubtleLabel")
        self._vis_metrics_label.setWordWrap(True)
        layout.addWidget(self._vis_metrics_label)

        self._display_group = QGroupBox()
        display_layout = QVBoxLayout(self._display_group)

        self._show_deformed = QCheckBox()
        self._show_deformed.setChecked(True)
        self._show_deformed.toggled.connect(self._mesh_canvas.set_show_deformed)
        display_layout.addWidget(self._show_deformed)

        self._show_constraints = QCheckBox()
        self._show_constraints.setChecked(True)
        self._show_constraints.toggled.connect(self._mesh_canvas.set_show_constraints)
        display_layout.addWidget(self._show_constraints)

        self._show_loads = QCheckBox()
        self._show_loads.setChecked(True)
        self._show_loads.toggled.connect(self._mesh_canvas.set_show_loads)
        display_layout.addWidget(self._show_loads)

        self._show_node_ids = QCheckBox()
        self._show_node_ids.toggled.connect(self._mesh_canvas.set_show_node_ids)
        display_layout.addWidget(self._show_node_ids)

        self._show_element_ids = QCheckBox()
        self._show_element_ids.toggled.connect(self._mesh_canvas.set_show_element_ids)
        display_layout.addWidget(self._show_element_ids)

        self._scale_row_label = QLabel()
        scale_row = QHBoxLayout()
        scale_row.addWidget(self._scale_row_label)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setDecimals(1)
        self._scale_spin.setRange(0.0, 1_000_000.0)
        self._scale_spin.setValue(300.0)
        self._scale_spin.valueChanged.connect(self._mesh_canvas.set_deformation_scale)
        scale_row.addWidget(self._scale_spin)
        display_layout.addLayout(scale_row)

        self._result_display_group = QGroupBox()
        result_layout = QFormLayout(self._result_display_group)

        self._show_contour = QCheckBox()
        self._show_contour.setChecked(True)
        self._show_contour.toggled.connect(self._mesh_canvas.set_show_contour)
        result_layout.addRow(self._show_contour)

        self._contour_label = QLabel()
        self._contour_combo = QComboBox()
        self._contour_combo.addItem("sx", "sx")
        self._contour_combo.addItem("sy", "sy")
        self._contour_combo.addItem("txy", "txy")
        self._contour_combo.addItem("ex", "ex")
        self._contour_combo.addItem("ey", "ey")
        self._contour_combo.addItem("gxy", "gxy")
        self._contour_combo.addItem("|u|", "u_mag")
        self._contour_combo.addItem("s1", "s1")
        self._contour_combo.addItem("mises", "mise")
        self._contour_combo.addItem("|F|", "load_mag")
        self._contour_combo.currentIndexChanged.connect(self._on_contour_variable_changed)
        result_layout.addRow(self._contour_label, self._contour_combo)

        layout.addWidget(self._display_group)
        layout.addWidget(self._result_display_group)
        layout.addStretch(1)
        return tab

    def _build_results_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self._results_state = QLabel()
        self._results_state.setObjectName("SubtleLabel")
        self._results_state.setWordWrap(True)
        layout.addWidget(self._results_state)

        scope_row = QHBoxLayout()
        self._results_scope_label = QLabel()
        scope_row.addWidget(self._results_scope_label)
        self._results_scope_type_combo = QComboBox()
        self._results_scope_type_combo.currentIndexChanged.connect(self._on_results_scope_type_changed)
        self._results_scope_type_combo.addItem("All", "all")
        self._results_scope_type_combo.addItem("Component", "component")
        self._results_scope_type_combo.addItem("Region", "region")
        self._results_scope_type_combo.addItem("Material", "material")
        self._results_scope_type_combo.addItem("Set", "set")
        scope_row.addWidget(self._results_scope_type_combo)
        self._results_scope_target_combo = QComboBox()
        self._results_scope_target_combo.currentIndexChanged.connect(self._on_results_scope_target_changed)
        scope_row.addWidget(self._results_scope_target_combo)
        self._btn_results_scope_clear = QPushButton()
        self._btn_results_scope_clear.clicked.connect(self._clear_results_scope_filter)
        scope_row.addWidget(self._btn_results_scope_clear)
        layout.addLayout(scope_row)

        self._results_tabs = QTabWidget()
        self._summary_text = QPlainTextEdit()
        self._summary_text.setReadOnly(True)
        self._results_tabs.addTab(self._summary_text, "")

        self._node_page = QWidget()
        node_layout = QVBoxLayout(self._node_page)
        node_row = QHBoxLayout()
        self._node_filter_label = QLabel()
        node_row.addWidget(self._node_filter_label)
        self._node_filter_edit = QLineEdit()
        self._node_filter_edit.textChanged.connect(self._refresh_node_table)
        node_row.addWidget(self._node_filter_edit)
        self._btn_export_nodes = QPushButton()
        self._btn_export_nodes.clicked.connect(self._export_node_table)
        node_row.addWidget(self._btn_export_nodes)
        node_layout.addLayout(node_row)
        self._node_selection_label = QLabel()
        self._node_selection_label.setObjectName("SubtleLabel")
        node_layout.addWidget(self._node_selection_label)
        self._node_table = QTableWidget(0, 3)
        self._node_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._node_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._node_table.setSortingEnabled(True)
        self._node_table.itemSelectionChanged.connect(self._on_node_selection_changed)
        node_layout.addWidget(self._node_table)
        self._results_tabs.addTab(self._node_page, "")

        self._element_page = QWidget()
        element_layout = QVBoxLayout(self._element_page)
        element_row = QHBoxLayout()
        self._element_filter_label = QLabel()
        element_row.addWidget(self._element_filter_label)
        self._element_filter_edit = QLineEdit()
        self._element_filter_edit.textChanged.connect(self._refresh_element_table)
        element_row.addWidget(self._element_filter_edit)
        self._btn_export_elements = QPushButton()
        self._btn_export_elements.clicked.connect(self._export_element_table)
        element_row.addWidget(self._btn_export_elements)
        element_layout.addLayout(element_row)
        self._element_selection_label = QLabel()
        self._element_selection_label.setObjectName("SubtleLabel")
        element_layout.addWidget(self._element_selection_label)
        self._element_table = QTableWidget(0, 7)
        self._element_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._element_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._element_table.setSortingEnabled(True)
        self._element_table.itemSelectionChanged.connect(self._on_element_selection_changed)
        element_layout.addWidget(self._element_table)
        self._results_tabs.addTab(self._element_page, "")

        layout.addWidget(self._results_tabs)
        return tab

    def _build_log_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self._log_info = QLabel()
        self._log_info.setObjectName("SubtleLabel")
        self._log_info.setWordWrap(True)
        layout.addWidget(self._log_info)
        self._btn_log_toggle_tab = QPushButton()
        self._btn_log_toggle_tab.clicked.connect(self._toggle_log_panel)
        layout.addWidget(self._btn_log_toggle_tab)
        self._btn_log_clear = QPushButton()
        self._btn_log_clear.clicked.connect(self._clear_logs)
        layout.addWidget(self._btn_log_clear)
        self._log_hint = QLabel()
        self._log_hint.setObjectName("SubtleLabel")
        layout.addWidget(self._log_hint)
        layout.addStretch(1)
        return tab

    def _build_log_dock(self) -> None:
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        dock_widget = QWidget()
        dock_layout = QVBoxLayout(dock_widget)
        dock_layout.setContentsMargins(6, 6, 6, 6)
        dock_layout.addWidget(self._log_text)
        self._log_dock = QDockWidget(self)
        self._log_dock.setWidget(dock_widget)
        self._log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self._log_dock.visibilityChanged.connect(self._on_log_visibility_changed)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
        self._log_dock.hide()

    def _import_ai_model_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model image",
            str(Path.cwd()),
            "Image (*.png *.jpg *.jpeg *.bmp)",
        )
        if not path:
            return
        self._rock_ai_image_path.setText(path)
        self._log(f"AI image entry prepared: {path}")
        self._set_runtime_state("ready", "AI image recognition entry connected (placeholder parser).")

    def _run_ai_context_model(self) -> None:
        text = self._rock_ai_context_text.toPlainText().strip()
        if not text:
            self._run_geotech_from_text_dialog()
            return
        try:
            template = build_template_input_from_text_description(
                text,
                backend=self._backend_combo.currentText(),
            )
        except Exception as exc:
            self._show_error("AI context parse failed", str(exc))
            return
        self._run_geotech_template(template)

    def _axis_seed_coords(
        self,
        length: float,
        global_seed: float,
        *,
        local_seed: float,
        use_local: bool,
        local_min: float,
        local_max: float,
    ) -> list[float]:
        coords = [0.0]
        x = 0.0
        safe_global = max(global_seed, 1e-6)
        safe_local = max(local_seed, 1e-6)
        lo = min(local_min, local_max)
        hi = max(local_min, local_max)
        while x < length - 1e-9:
            if use_local and lo <= x <= hi:
                step = safe_local
            else:
                step = safe_global
            x = min(length, x + step)
            if x - coords[-1] <= 1e-9:
                break
            coords.append(float(x))
        if coords[-1] < length:
            coords.append(float(length))
        return coords

    def _build_grid_model(self, coords_x: list[float], coords_y: list[float], material: Material) -> Model:
        nodes: list[Node] = []
        node_id = 1
        node_id_by_ij: dict[tuple[int, int], int] = {}
        for j, y in enumerate(coords_y):
            for i, x in enumerate(coords_x):
                nodes.append(Node(id=node_id, x=float(x), y=float(y)))
                node_id_by_ij[(i, j)] = node_id
                node_id += 1

        elements: list[Element] = []
        element_id = 1
        for j in range(len(coords_y) - 1):
            for i in range(len(coords_x) - 1):
                n1 = node_id_by_ij[(i, j)]
                n2 = node_id_by_ij[(i + 1, j)]
                n3 = node_id_by_ij[(i, j + 1)]
                n4 = node_id_by_ij[(i + 1, j + 1)]
                elements.append(
                    Element(
                        id=element_id,
                        type="T3",
                        connectivity=[n1, n2, n4],
                        material_id=material.id,
                    )
                )
                element_id += 1
                elements.append(
                    Element(
                        id=element_id,
                        type="T3",
                        connectivity=[n1, n4, n3],
                        material_id=material.id,
                    )
                )
                element_id += 1

        return Model(
            mesh=Mesh(nodes=nodes, elements=elements),
            materials=[material],
        )

    def _create_rect_model_from_form(self) -> None:
        width = float(self._rock_rect_width.value())
        height = float(self._rock_rect_height.value())
        seed = float(self._rock_rect_seed.value())
        self._active_sketch_width = width
        self._active_sketch_height = height
        self._active_sketch_grid_step = seed
        self._sync_part_shape_controls_from_size(max(width * 0.35, 1e-3), max(height * 0.35, 1e-3))
        self._clear_mapping_state()
        self._reset_scene_project()
        self._part_sketch_points.clear()
        self._part_sketch_history.clear()
        self._part_sketch_curve_hint = None
        self._part_sketch_undo_stack.clear()
        self._part_sketch_redo_stack.clear()
        self._material_geometry_region_rules.clear()
        self._selected_face_region_sketch_keys.clear()
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._set_model(Model(mesh=Mesh()))
        self._mesh_canvas.set_sketch_plane(width, height, seed)
        self._editor_canvas_edit_enable.setChecked(True)
        idx = self._editor_canvas_tool_combo.findData("add_node")
        if idx >= 0:
            self._editor_canvas_tool_combo.setCurrentIndex(idx)
        self._capture_sketch_from_canvas = True
        self._part_capture_from_canvas.setChecked(True)
        self._pending_pick_context = None
        self._model_measure_result.setText(self._ui("测距: 等待选择两点", "Measure: pick two points"))
        self._log(
            self._ui(
                f"创建二维草图平面完成: 宽={width:.3f}, 高={height:.3f}, 网格步长={seed:.3f}。",
                f"2D sketch plane created: width={width:.3f}, height={height:.3f}, grid_step={seed:.3f}.",
            )
        )

    def _snap_sketch_point(self, x: float, y: float) -> tuple[float, float]:
        sx = float(x)
        sy = float(y)
        if getattr(self, "_part_snap_to_grid", None) is not None and self._part_snap_to_grid.isChecked():
            step = max(float(self._active_sketch_grid_step), 1e-6)
            sx = round(sx / step) * step
            sy = round(sy / step) * step

        sx = max(0.0, min(float(self._active_sketch_width), sx))
        sy = max(0.0, min(float(self._active_sketch_height), sy))

        if self._part_sketch_points:
            tol = max(self._active_sketch_grid_step * 0.35, 0.02)
            for px, py in self._part_sketch_points:
                if (sx - px) * (sx - px) + (sy - py) * (sy - py) <= tol * tol:
                    sx, sy = px, py
                    break
        return (sx, sy)

    def _add_part_sketch_point(self) -> None:
        point = self._snap_sketch_point(float(self._part_point_x.value()), float(self._part_point_y.value()))
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._selected_part_segment = None
        self._selected_part_angle = None
        self._part_sketch_points.append(point)
        self._part_point_x.setValue(point[0])
        self._part_point_y.setValue(point[1])
        self._refresh_part_sketch_table()

    def _selected_part_sketch_rows(self) -> list[int]:
        selection_model = self._part_sketch_table.selectionModel()
        if selection_model is None:
            return []
        rows = sorted({idx.row() for idx in selection_model.selectedRows(0)})
        return [row for row in rows if 0 <= row < len(self._part_sketch_points)]

    def _reset_part_sketch_selection(self) -> None:
        self._selected_part_segment = None
        self._selected_part_angle = None
        self._selected_face_region_sketch_keys.clear()
        self._selected_face_region_element_ids.clear()
        self._selected_geometry_point_ids.clear()
        self._selected_geometry_edge_ids.clear()
        self._scene_selection.active_region_ids.clear()
        self._mesh_canvas.set_selected_sketch_segment(None)
        self._mesh_canvas.set_highlighted_sketch_faces([])
        selection_model = self._part_sketch_table.selectionModel()
        if selection_model is not None:
            selection_model.clearSelection()

    def _select_part_sketch_rows(self, rows: list[int]) -> None:
        selection_model = self._part_sketch_table.selectionModel()
        if selection_model is None:
            return
        self._suppress_part_selection_changed = True
        try:
            selection_model.clearSelection()
            for row in rows:
                if 0 <= row < self._part_sketch_table.rowCount():
                    index = self._part_sketch_table.model().index(row, 0)
                    selection_model.select(
                        index,
                        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                    )
        finally:
            self._suppress_part_selection_changed = False

    def _on_canvas_sketch_segment_picked(self, i0: int, i1: int) -> None:
        self._selected_part_segment = (int(i0), int(i1))
        self._selected_part_angle = None
        self._select_part_sketch_rows([int(i0), int(i1)])
        idx = self._part_quick_mode.findData("length")
        if idx >= 0:
            self._part_quick_mode.setCurrentIndex(idx)
        self._mesh_canvas.set_selected_sketch_segment((int(i0), int(i1)))
        self._part_dimension_status.setText(
            self._tr("rock.part.selection.segment", "已选中线段（顺序生效：第一点固定，第二点移动）。")
        )
        self._update_quick_dimension_apply_state()

    def _on_canvas_sketch_angle_picked(self, ia: int, ib: int, ic: int) -> None:
        self._selected_part_angle = (int(ia), int(ib), int(ic))
        self._selected_part_segment = None
        self._select_part_sketch_rows([int(ia), int(ib), int(ic)])
        idx = self._part_quick_mode.findData("angle")
        if idx >= 0:
            self._part_quick_mode.setCurrentIndex(idx)
        self._mesh_canvas.set_selected_sketch_angle((int(ia), int(ib), int(ic)))
        self._part_dimension_status.setText(
            self._tr("rock.part.selection.angle", "已选中夹角（第一边固定，第二边旋转）。")
        )
        self._update_quick_dimension_apply_state()

    def _on_canvas_sketch_history_segment_picked(self, history_index: int, i0: int, i1: int) -> None:
        if not (0 <= int(history_index) < len(self._part_sketch_history)):
            return
        self._push_part_undo_snapshot()
        selected = self._part_sketch_history.pop(int(history_index))
        self._archive_current_sketch_to_history()

        points_raw = selected.get("points", [])
        hint_raw = selected.get("hint")
        points = [(float(x), float(y)) for x, y in points_raw] if isinstance(points_raw, list) else []
        if len(points) < 2:
            return
        self._part_sketch_points = points
        self._part_sketch_curve_hint = copy.deepcopy(hint_raw) if isinstance(hint_raw, dict) else None
        self._selected_part_angle = None
        self._selected_part_segment = None
        self._refresh_part_sketch_table()

        if 0 <= int(i0) < len(self._part_sketch_points) and 0 <= int(i1) < len(self._part_sketch_points):
            self._on_canvas_sketch_segment_picked(int(i0), int(i1))
        self._log(
            self._ui(
                "已激活历史图形，可继续测量和标定。",
                "History shape activated for measurement and dimension edit.",
            )
        )

    def _on_part_sketch_selection_changed(self) -> None:
        if self._suppress_part_selection_changed:
            return
        rows = self._selected_part_sketch_rows()
        count = len(rows)
        if count == 2:
            self._selected_part_segment = (rows[0], rows[1])
            self._selected_part_angle = None
            self._mesh_canvas.set_selected_sketch_segment(self._selected_part_segment)
            self._part_dimension_status.setText(
                self._tr("rock.part.selection.two", "已选择2个点，可应用长度标定。")
            )
        elif count == 3:
            self._selected_part_angle = (rows[0], rows[1], rows[2])
            self._selected_part_segment = None
            self._mesh_canvas.set_selected_sketch_angle(self._selected_part_angle)
            self._part_dimension_status.setText(
                self._tr("rock.part.selection.three", "已选择3个点，可应用角度标定（A-顶点B-C）。")
            )
        else:
            self._selected_part_segment = None
            self._selected_part_angle = None
            self._mesh_canvas.set_selected_sketch_segment(None)
            self._part_dimension_status.setText(
                self._tr("rock.part.dimension.hint", "提示：长度标定选2点，角度标定按 A-顶点B-C 选3点。")
            )
        self._update_quick_dimension_apply_state()

    def _on_part_quick_mode_changed(self, _: int) -> None:
        self._update_quick_dimension_apply_state()

    def _update_quick_dimension_apply_state(self) -> None:
        if not hasattr(self, "_btn_part_quick_apply"):
            return
        mode = self._part_quick_mode.currentData()
        if mode == "angle":
            enabled = self._selected_part_angle is not None or len(self._selected_part_sketch_rows()) == 3
            self._btn_part_quick_apply.setEnabled(enabled)
            return
        enabled = self._selected_part_segment is not None or len(self._selected_part_sketch_rows()) == 2
        self._btn_part_quick_apply.setEnabled(enabled)

    def _is_part_sketch_closed(self) -> bool:
        if len(self._part_sketch_points) < 2:
            return False
        first = self._part_sketch_points[0]
        last = self._part_sketch_points[-1]
        return abs(first[0] - last[0]) <= 1e-9 and abs(first[1] - last[1]) <= 1e-9

    def _copy_part_sketch_points(self) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self._part_sketch_points]

    def _copy_part_sketch_history(self) -> list[dict[str, object]]:
        copied: list[dict[str, object]] = []
        for item in self._part_sketch_history:
            points_raw = item.get("points")
            points = [(float(x), float(y)) for x, y in points_raw] if isinstance(points_raw, list) else []
            hint = copy.deepcopy(item.get("hint"))
            copied.append({"points": points, "hint": hint})
        return copied

    def _copy_part_sketch_state(self) -> dict[str, object]:
        return {
            "points": self._copy_part_sketch_points(),
            "history": self._copy_part_sketch_history(),
            "hint": copy.deepcopy(self._part_sketch_curve_hint),
        }

    def _current_part_sketch_geometry_signature(self) -> str:
        chunks: list[str] = []
        for x, y in self._part_sketch_points:
            chunks.append(f"P:{float(x):.9g},{float(y):.9g}")
        chunks.append("|")
        for idx, item in enumerate(self._part_sketch_history):
            chunks.append(f"H{idx}")
            points_raw = item.get("points", [])
            if isinstance(points_raw, list):
                for x, y in points_raw:
                    chunks.append(f"{float(x):.9g},{float(y):.9g}")
            hint = item.get("hint")
            if isinstance(hint, dict):
                kind = str(hint.get("kind", ""))
                bbox = hint.get("bbox")
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    chunks.append(
                        f"K:{kind}:{float(bbox[0]):.9g},{float(bbox[1]):.9g},{float(bbox[2]):.9g},{float(bbox[3]):.9g}"
                    )
            chunks.append(";")
        if isinstance(self._part_sketch_curve_hint, dict):
            kind = str(self._part_sketch_curve_hint.get("kind", ""))
            bbox = self._part_sketch_curve_hint.get("bbox")
            if isinstance(bbox, tuple) and len(bbox) == 4:
                chunks.append(
                    f"C:{kind}:{float(bbox[0]):.9g},{float(bbox[1]):.9g},{float(bbox[2]):.9g},{float(bbox[3]):.9g}"
                )
        return "|".join(chunks)

    def _restore_part_sketch_state(self, state: dict[str, object]) -> None:
        points_raw = state.get("points", [])
        history_raw = state.get("history", [])
        hint_raw = state.get("hint")
        self._part_sketch_points = [(float(x), float(y)) for x, y in points_raw] if isinstance(points_raw, list) else []
        self._part_sketch_history = []
        if isinstance(history_raw, list):
            for item in history_raw:
                if not isinstance(item, dict):
                    continue
                points = item.get("points", [])
                hint = item.get("hint")
                self._part_sketch_history.append(
                    {
                        "points": [(float(x), float(y)) for x, y in points] if isinstance(points, list) else [],
                        "hint": copy.deepcopy(hint),
                    }
                )
        self._part_sketch_curve_hint = copy.deepcopy(hint_raw) if isinstance(hint_raw, dict) else None

    def _clear_part_curve_hint(self) -> None:
        self._part_sketch_curve_hint = None

    def _archive_current_sketch_to_history(self) -> None:
        if len(self._part_sketch_points) < 2:
            return
        self._part_sketch_history.append(
            {
                "points": self._copy_part_sketch_points(),
                "hint": copy.deepcopy(self._part_sketch_curve_hint),
            }
        )
        if len(self._part_sketch_history) > 80:
            self._part_sketch_history = self._part_sketch_history[-80:]
        self._part_sketch_curve_hint = None

    @staticmethod
    def _build_curve_hint(kind: str, x0: float, y0: float, x1: float, y1: float) -> dict[str, object]:
        min_x = min(float(x0), float(x1))
        max_x = max(float(x0), float(x1))
        min_y = min(float(y0), float(y1))
        max_y = max(float(y0), float(y1))
        return {
            "kind": str(kind),
            "bbox": (min_x, min_y, max_x, max_y),
            "center": (0.5 * (min_x + max_x), 0.5 * (min_y + max_y)),
        }

    def _sync_part_shape_controls_from_size(self, width: float, height: float) -> None:
        if not hasattr(self, "_part_shape_w") or not hasattr(self, "_part_shape_h"):
            return
        self._updating_part_shape_controls = True
        try:
            self._part_shape_w.setValue(max(float(width), 1e-6))
            self._part_shape_h.setValue(max(float(height), 1e-6))
        finally:
            self._updating_part_shape_controls = False
        self._part_shape_ratio = self._part_shape_w.value() / max(self._part_shape_h.value(), 1e-6)

    def _on_part_shape_lock_toggled(self, checked: bool) -> None:
        if checked:
            self._part_shape_ratio = self._part_shape_w.value() / max(self._part_shape_h.value(), 1e-6)

    def _on_part_shape_w_changed(self, value: float) -> None:
        if self._updating_part_shape_controls:
            return
        self._active_sketch_width = max(float(value), 1e-6)
        if hasattr(self, "_part_shape_lock_ratio") and self._part_shape_lock_ratio.isChecked():
            ratio = max(float(self._part_shape_ratio), 1e-9)
            new_h = max(float(value) / ratio, 1e-6)
            self._updating_part_shape_controls = True
            try:
                self._part_shape_h.setValue(new_h)
            finally:
                self._updating_part_shape_controls = False
            self._active_sketch_height = new_h
        else:
            self._part_shape_ratio = self._part_shape_w.value() / max(self._part_shape_h.value(), 1e-6)

    def _on_part_shape_h_changed(self, value: float) -> None:
        if self._updating_part_shape_controls:
            return
        self._active_sketch_height = max(float(value), 1e-6)
        if hasattr(self, "_part_shape_lock_ratio") and self._part_shape_lock_ratio.isChecked():
            ratio = max(float(self._part_shape_ratio), 1e-9)
            new_w = max(float(value) * ratio, 1e-6)
            self._updating_part_shape_controls = True
            try:
                self._part_shape_w.setValue(new_w)
            finally:
                self._updating_part_shape_controls = False
            self._active_sketch_width = new_w
        else:
            self._part_shape_ratio = self._part_shape_w.value() / max(self._part_shape_h.value(), 1e-6)

    def _push_part_undo_snapshot(self) -> None:
        self._part_sketch_undo_stack.append(self._copy_part_sketch_state())
        # Keep bounded history for stable memory usage in long sessions.
        if len(self._part_sketch_undo_stack) > 300:
            self._part_sketch_undo_stack = self._part_sketch_undo_stack[-300:]
        self._part_sketch_redo_stack.clear()

    def _preserve_closed_loop_endpoint(self, updated_index: int, closed_before: bool) -> None:
        if not closed_before or len(self._part_sketch_points) < 2:
            return
        last_idx = len(self._part_sketch_points) - 1
        if updated_index == 0:
            self._part_sketch_points[last_idx] = self._part_sketch_points[0]
        elif updated_index == last_idx:
            self._part_sketch_points[0] = self._part_sketch_points[last_idx]

    def _fill_length_from_measurement(self) -> None:
        text = self._model_measure_result.text().strip()
        if "L" not in text:
            self._show_error(
                self._ui("长度提取失败", "Length extraction failed"),
                self._ui("当前没有可用的测距结果，请先使用测量工具。", "No measurement result found. Use measure tool first."),
            )
            return
        try:
            # Supports "测距: L = ..." and "Measure: L = ..."
            value_text = text.split("=", maxsplit=1)[1].strip()
            value = float(value_text)
        except Exception:
            self._show_error(
                self._ui("长度提取失败", "Length extraction failed"),
                self._ui("无法从测距结果解析长度。", "Failed to parse length from measurement."),
            )
            return
        self._part_length_target.setValue(max(1e-6, value))
        self._part_dimension_status.setText(
            self._ui(f"已填入测距长度: {value:.6g}", f"Measured length loaded: {value:.6g}")
        )

    def _apply_part_length_constraint(self) -> None:
        if self._selected_part_segment is not None:
            rows = [self._selected_part_segment[0], self._selected_part_segment[1]]
        else:
            rows = self._selected_part_sketch_rows()
        if len(rows) != 2:
            self._show_error(
                self._ui("长度标定失败", "Length calibration failed"),
                self._ui("请在草图表中选中 2 个点（起点、终点）。", "Select exactly 2 sketch points (start, end)."),
            )
            return
        i0, i1 = rows[0], rows[1]
        p0 = self._part_sketch_points[i0]
        p1 = self._part_sketch_points[i1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = hypot(dx, dy)
        if length <= 1e-12:
            self._show_error(
                self._ui("长度标定失败", "Length calibration failed"),
                self._ui("两点重合，无法定义方向。", "Selected points are coincident; direction is undefined."),
            )
            return
        target = float(self._part_length_target.value())
        scale = target / length
        new_p1 = (p0[0] + dx * scale, p0[1] + dy * scale)
        closed_before = self._is_part_sketch_closed()
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._part_sketch_points[i1] = new_p1
        self._preserve_closed_loop_endpoint(i1, closed_before)
        self._refresh_part_sketch_table()
        self._part_dimension_status.setText(
            self._ui(
                f"长度标定完成: P{i0+1}-P{i1+1} = {target:.6g}",
                f"Length calibrated: P{i0+1}-P{i1+1} = {target:.6g}",
            )
        )

    def _apply_part_angle_constraint(self) -> None:
        if self._selected_part_angle is not None:
            rows = [self._selected_part_angle[0], self._selected_part_angle[1], self._selected_part_angle[2]]
        else:
            rows = self._selected_part_sketch_rows()
        if len(rows) != 3:
            self._show_error(
                self._ui("角度标定失败", "Angle calibration failed"),
                self._ui("请在草图表中选中 3 个点（A-顶点B-C）。", "Select exactly 3 points in order (A-vertex B-C)."),
            )
            return
        ia, ib, ic = rows[0], rows[1], rows[2]
        pa = self._part_sketch_points[ia]
        pb = self._part_sketch_points[ib]
        pc = self._part_sketch_points[ic]

        v1x = pa[0] - pb[0]
        v1y = pa[1] - pb[1]
        v2x = pc[0] - pb[0]
        v2y = pc[1] - pb[1]
        len1 = hypot(v1x, v1y)
        len2 = hypot(v2x, v2y)
        if len1 <= 1e-12 or len2 <= 1e-12:
            self._show_error(
                self._ui("角度标定失败", "Angle calibration failed"),
                self._ui("点A/B/C中存在重合，无法定义角度。", "Coincident points found; angle cannot be defined."),
            )
            return

        base_dir = float(atan2(v1y, v1x))
        target_deg = float(self._part_angle_target.value())
        target_rad = float(radians(target_deg))
        cross = v1x * v2y - v1y * v2x
        sign = 1.0 if cross >= 0.0 else -1.0
        new_dir = base_dir + sign * target_rad
        new_pc = (
            pb[0] + len2 * float(cos(new_dir)),
            pb[1] + len2 * float(sin(new_dir)),
        )

        closed_before = self._is_part_sketch_closed()
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._part_sketch_points[ic] = new_pc
        self._preserve_closed_loop_endpoint(ic, closed_before)
        self._refresh_part_sketch_table()
        self._part_dimension_status.setText(
            self._ui(
                f"角度标定完成: ∠(P{ia+1},P{ib+1},P{ic+1}) = {target_deg:.3f}°",
                f"Angle calibrated: ∠(P{ia+1},P{ib+1},P{ic+1}) = {target_deg:.3f}°",
            )
        )

    def _resolve_single_segment_for_constraint(self) -> tuple[int, int] | None:
        if self._selected_part_segment is not None:
            i0, i1 = self._selected_part_segment
            return (int(i0), int(i1))
        rows = self._selected_part_sketch_rows()
        if len(rows) == 2:
            return (rows[0], rows[1])
        return None

    def _resolve_two_segments_for_constraint(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        rows = self._selected_part_sketch_rows()
        if len(rows) == 4:
            return ((rows[0], rows[1]), (rows[2], rows[3]))
        return None

    def _apply_segment_horizontal_constraint(self) -> None:
        pair = self._resolve_single_segment_for_constraint()
        if pair is None:
            self._show_error(
                self._ui("约束失败", "Constraint failed"),
                self._ui("请先选择一个线段（2个点）。", "Select one segment first (2 points)."),
            )
            return
        i0, i1 = pair
        p0 = self._part_sketch_points[i0]
        closed_before = self._is_part_sketch_closed()
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._part_sketch_points[i1] = (self._part_sketch_points[i1][0], p0[1])
        self._preserve_closed_loop_endpoint(i1, closed_before)
        self._refresh_part_sketch_table()
        self._part_dimension_status.setText(self._tr("rock.part.constraint.horizontal.done", "已应用水平约束。"))

    def _apply_segment_vertical_constraint(self) -> None:
        pair = self._resolve_single_segment_for_constraint()
        if pair is None:
            self._show_error(
                self._ui("约束失败", "Constraint failed"),
                self._ui("请先选择一个线段（2个点）。", "Select one segment first (2 points)."),
            )
            return
        i0, i1 = pair
        p0 = self._part_sketch_points[i0]
        closed_before = self._is_part_sketch_closed()
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._part_sketch_points[i1] = (p0[0], self._part_sketch_points[i1][1])
        self._preserve_closed_loop_endpoint(i1, closed_before)
        self._refresh_part_sketch_table()
        self._part_dimension_status.setText(self._tr("rock.part.constraint.vertical.done", "已应用垂直约束。"))

    def _apply_points_collinear_constraint(self) -> None:
        rows = self._selected_part_sketch_rows()
        if len(rows) != 3:
            self._show_error(
                self._ui("约束失败", "Constraint failed"),
                self._ui("共线约束需要选择3个点。", "Collinear constraint requires 3 points."),
            )
            return
        ia, ib, ic = rows
        pa = self._part_sketch_points[ia]
        pb = self._part_sketch_points[ib]
        pc = self._part_sketch_points[ic]
        vx = pb[0] - pa[0]
        vy = pb[1] - pa[1]
        vv = vx * vx + vy * vy
        if vv <= 1e-14:
            self._show_error(
                self._ui("约束失败", "Constraint failed"),
                self._ui("前两点重合，无法定义共线方向。", "First two points are coincident; line direction is undefined."),
            )
            return
        t = ((pc[0] - pa[0]) * vx + (pc[1] - pa[1]) * vy) / vv
        projected = (pa[0] + t * vx, pa[1] + t * vy)
        closed_before = self._is_part_sketch_closed()
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._part_sketch_points[ic] = projected
        self._preserve_closed_loop_endpoint(ic, closed_before)
        self._refresh_part_sketch_table()
        self._part_dimension_status.setText(self._tr("rock.part.constraint.collinear.done", "已应用共线约束。"))

    def _apply_segments_parallel_constraint(self) -> None:
        pair = self._resolve_two_segments_for_constraint()
        if pair is None:
            self._show_error(
                self._ui("约束失败", "Constraint failed"),
                self._ui("平行约束需要选择4个点（两条线段）。", "Parallel constraint requires 4 points (two segments)."),
            )
            return
        (a0, a1), (b0, b1) = pair
        p0 = self._part_sketch_points[a0]
        p1 = self._part_sketch_points[a1]
        q0 = self._part_sketch_points[b0]
        q1 = self._part_sketch_points[b1]
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        vnorm = hypot(vx, vy)
        qlen = hypot(q1[0] - q0[0], q1[1] - q0[1])
        if vnorm <= 1e-12 or qlen <= 1e-12:
            self._show_error(self._ui("约束失败", "Constraint failed"), self._ui("线段长度过小，无法施加平行约束。", "Segment too short for parallel constraint."))
            return
        ux, uy = vx / vnorm, vy / vnorm
        dot = (q1[0] - q0[0]) * ux + (q1[1] - q0[1]) * uy
        sign = 1.0 if dot >= 0.0 else -1.0
        new_q1 = (q0[0] + sign * qlen * ux, q0[1] + sign * qlen * uy)
        closed_before = self._is_part_sketch_closed()
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._part_sketch_points[b1] = new_q1
        self._preserve_closed_loop_endpoint(b1, closed_before)
        self._refresh_part_sketch_table()
        self._part_dimension_status.setText(self._tr("rock.part.constraint.parallel.done", "已应用平行约束。"))

    def _apply_segments_perpendicular_constraint(self) -> None:
        pair = self._resolve_two_segments_for_constraint()
        if pair is None:
            self._show_error(
                self._ui("约束失败", "Constraint failed"),
                self._ui("垂直约束需要选择4个点（两条线段）。", "Perpendicular constraint requires 4 points (two segments)."),
            )
            return
        (a0, a1), (b0, b1) = pair
        p0 = self._part_sketch_points[a0]
        p1 = self._part_sketch_points[a1]
        q0 = self._part_sketch_points[b0]
        q1 = self._part_sketch_points[b1]
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        vnorm = hypot(vx, vy)
        qlen = hypot(q1[0] - q0[0], q1[1] - q0[1])
        if vnorm <= 1e-12 or qlen <= 1e-12:
            self._show_error(self._ui("约束失败", "Constraint failed"), self._ui("线段长度过小，无法施加垂直约束。", "Segment too short for perpendicular constraint."))
            return
        ux, uy = -vy / vnorm, vx / vnorm
        dot = (q1[0] - q0[0]) * ux + (q1[1] - q0[1]) * uy
        sign = 1.0 if dot >= 0.0 else -1.0
        new_q1 = (q0[0] + sign * qlen * ux, q0[1] + sign * qlen * uy)
        closed_before = self._is_part_sketch_closed()
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._part_sketch_points[b1] = new_q1
        self._preserve_closed_loop_endpoint(b1, closed_before)
        self._refresh_part_sketch_table()
        self._part_dimension_status.setText(self._tr("rock.part.constraint.perpendicular.done", "已应用垂直约束。"))

    def _undo_last_sketch_point(self) -> None:
        """Undo last Part sketch operation (point insert or geometry edit)."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QPlainTextEdit)):
            return
        if not self._part_sketch_undo_stack:
            return
        current = self._copy_part_sketch_state()
        previous = self._part_sketch_undo_stack.pop()
        self._part_sketch_redo_stack.append(current)
        self._restore_part_sketch_state(previous)
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._log(
            self._ui(
                f"草图撤销: 恢复到上一步（当前点数 {len(self._part_sketch_points)}）",
                f"Sketch undo: reverted one operation (points={len(self._part_sketch_points)}).",
            )
        )

    def _redo_last_sketch_point(self) -> None:
        """Redo last undone Part sketch operation."""
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QPlainTextEdit)):
            return
        if not self._part_sketch_redo_stack:
            return
        current = self._copy_part_sketch_state()
        restored = self._part_sketch_redo_stack.pop()
        self._part_sketch_undo_stack.append(current)
        self._restore_part_sketch_state(restored)
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._log(
            self._ui(
                f"草图重做: 前进一步（当前点数 {len(self._part_sketch_points)}）",
                f"Sketch redo: advanced one operation (points={len(self._part_sketch_points)}).",
            )
        )

    def _close_part_sketch_loop(self) -> None:
        if len(self._part_sketch_points) < 3:
            return
        first = self._part_sketch_points[0]
        last = self._part_sketch_points[-1]
        if abs(first[0] - last[0]) <= 1e-9 and abs(first[1] - last[1]) <= 1e-9:
            return
        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        self._selected_part_segment = None
        self._selected_part_angle = None
        self._part_sketch_points.append(first)
        self._refresh_part_sketch_table()

    def _refresh_part_sketch_table(self) -> None:
        current_signature = self._current_part_sketch_geometry_signature()
        geometry_changed = current_signature != self._part_sketch_geometry_signature
        self._part_sketch_geometry_signature = current_signature
        if geometry_changed and self._material_geometry_region_rules:
            self._material_geometry_region_rules.clear()
            self._selected_face_region_sketch_keys.clear()
            self._mesh_canvas.set_highlighted_sketch_faces([])
            for region in self._scene_project.regions.values():
                if region.source == "sketch":
                    region.material_id = None
            if hasattr(self, "_material_face_hint"):
                self._material_face_hint.setText(
                    self._ui(
                        "草图已变更，已清空几何分配映射，请重新选择面区域。",
                        "Sketch changed; cleared geometry-material mappings. Please pick face again.",
                    )
                )
        self._part_sketch_table.setRowCount(len(self._part_sketch_points))
        for row, (x, y) in enumerate(self._part_sketch_points, start=1):
            self._part_sketch_table.setItem(row - 1, 0, QTableWidgetItem(str(row)))
            self._part_sketch_table.setItem(row - 1, 1, QTableWidgetItem(f"{x:.6g}"))
            self._part_sketch_table.setItem(row - 1, 2, QTableWidgetItem(f"{y:.6g}"))
        self._mesh_canvas.set_sketch_points(self._part_sketch_points)
        self._mesh_canvas.set_sketch_history(self._part_sketch_history)
        self._mesh_canvas.set_sketch_curve_hint(self._part_sketch_curve_hint)
        self._sync_material_sketch_faces_to_canvas()
        if self._selected_part_segment is not None:
            i0, i1 = self._selected_part_segment
            if 0 <= i0 < len(self._part_sketch_points) and 0 <= i1 < len(self._part_sketch_points):
                self._select_part_sketch_rows([i0, i1])
        elif self._selected_part_angle is not None:
            ia, ib, ic = self._selected_part_angle
            if 0 <= ia < len(self._part_sketch_points) and 0 <= ib < len(self._part_sketch_points) and 0 <= ic < len(self._part_sketch_points):
                self._select_part_sketch_rows([ia, ib, ic])
        if hasattr(self, "_btn_quick_undo"):
            self._btn_quick_undo.setEnabled(bool(self._part_sketch_undo_stack))
        if hasattr(self, "_btn_quick_redo"):
            self._btn_quick_redo.setEnabled(bool(self._part_sketch_redo_stack))
        self._sync_scene_from_sketch_geometry()
        self._update_sketch_face_highlight()
        self._update_quick_dimension_apply_state()

    def _clear_part_sketch(self) -> None:
        if self._part_sketch_points:
            self._push_part_undo_snapshot()
        self._part_sketch_points.clear()
        self._part_sketch_curve_hint = None
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()

    def _delete_part_sketch_rows(self, rows: list[int]) -> bool:
        valid_rows = sorted({int(row) for row in rows if 0 <= int(row) < len(self._part_sketch_points)})
        if not valid_rows:
            return False
        closed_before = self._is_part_sketch_closed()
        if closed_before and (0 in valid_rows or (len(self._part_sketch_points) - 1) in valid_rows):
            valid_rows = sorted(set(valid_rows + [0, len(self._part_sketch_points) - 1]))

        self._push_part_undo_snapshot()
        self._clear_part_curve_hint()
        for row in reversed(valid_rows):
            del self._part_sketch_points[row]

        if closed_before and len(self._part_sketch_points) >= 3:
            first = self._part_sketch_points[0]
            while len(self._part_sketch_points) >= 2:
                last = self._part_sketch_points[-1]
                if abs(first[0] - last[0]) <= 1e-9 and abs(first[1] - last[1]) <= 1e-9:
                    self._part_sketch_points.pop()
                else:
                    break
            self._part_sketch_points.append(self._part_sketch_points[0])

        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        return True

    def _clear_selected_part_geometry(self) -> None:
        rows = self._selected_part_sketch_rows()
        if rows:
            removed = self._delete_part_sketch_rows(rows)
            if removed:
                self._log(
                    self._ui(
                        f"已删除选中草图点 {len(rows)} 个。",
                        f"Removed {len(rows)} selected sketch points.",
                    )
                )
            return

        if self._part_sketch_points:
            self._push_part_undo_snapshot()
            self._part_sketch_points.clear()
            self._clear_part_curve_hint()
            self._reset_part_sketch_selection()
            self._refresh_part_sketch_table()
            self._log(self._ui("已清除当前草图图形。", "Active sketch geometry cleared."))
            return

        if self._part_sketch_history:
            self._push_part_undo_snapshot()
            self._part_sketch_history.pop()
            self._refresh_part_sketch_table()
            self._log(self._ui("已清除最近一个历史图形。", "Most recent history shape cleared."))

    def _clear_all_part_geometry(self) -> None:
        if not self._part_sketch_points and not self._part_sketch_history:
            return
        self._push_part_undo_snapshot()
        self._part_sketch_points.clear()
        self._part_sketch_history.clear()
        self._clear_part_curve_hint()
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._log(self._ui("已清除全部草图图形。", "All sketch geometry cleared."))

    def _on_capture_sketch_toggled(self, enabled: bool) -> None:
        self._capture_sketch_from_canvas = bool(enabled)
        self._editor_canvas_edit_enable.setChecked(bool(enabled))
        self._pending_pick_context = None
        if enabled:
            index = self._editor_canvas_tool_combo.findData("add_node")
            if index >= 0:
                self._editor_canvas_tool_combo.setCurrentIndex(index)
            self._show_model_group(self._rock_part_group)
            self._log(self._ui("已启用画布草图点录入，请在网格上点击点。", "Canvas sketch capture enabled. Click on grid to add points."))
        else:
            self._log(self._ui("已关闭画布草图点录入。", "Canvas sketch capture disabled."))

    def _build_model_from_polygon_points(self, points: list[tuple[float, float]]) -> Model:
        cleaned = list(points)
        if len(cleaned) >= 2:
            first = cleaned[0]
            last = cleaned[-1]
            if abs(first[0] - last[0]) <= 1e-12 and abs(first[1] - last[1]) <= 1e-12:
                cleaned = cleaned[:-1]
        if len(cleaned) < 3:
            raise ValueError("At least 3 non-collinear points are required.")

        material_id = int(self._material_assign_id_spin.value())
        material = Material(
            id=material_id,
            young_modulus=float(self._material_e_spin.value()),
            poisson_ratio=float(self._material_nu_spin.value()),
            plane_stress=bool(self._material_plane_stress.isChecked()),
        )

        nodes = [Node(id=i + 1, x=xy[0], y=xy[1]) for i, xy in enumerate(cleaned)]
        triangles = triangulate_polygon_ear_clipping(cleaned)
        elements = [
            Element(
                id=index + 1,
                type="T3",
                connectivity=[tri[0] + 1, tri[1] + 1, tri[2] + 1],
                material_id=material.id,
            )
            for index, tri in enumerate(triangles)
        ]
        return Model(
            mesh=Mesh(nodes=nodes, elements=elements),
            materials=[material],
        )

    def _requires_constrained_geometry_mesh(self) -> bool:
        sketch_loop_count = len([loop for loop in self._scene_project.loops.values() if loop.source == "sketch"])
        if sketch_loop_count > 1:
            return True
        if any(
            point.source == "manual" and point.role in {"edge_point", "interior_point"}
            for point in self._scene_project.geometry_points.values()
        ):
            return True
        if any(
            item.binding_mode == "geometry" and item.entity_type in {"point", "edge"} and item.entity_ids
            for item in self._scene_project.geometry_sets.values()
        ):
            return True
        return False

    def _build_model_from_scene_constrained(self) -> Model:
        sketch_loops = [loop for loop in self._scene_project.loops.values() if loop.source == "sketch"]
        if not sketch_loops:
            raise ValueError("No sketch loops available for constrained meshing.")
        loop_polygons: dict[str, list[tuple[float, float]]] = {}
        loop_areas: dict[str, float] = {}
        for loop in sketch_loops:
            points = self._normalize_polygon_points(loop.points)
            if len(points) < 3:
                continue
            loop_polygons[loop.id] = points
            loop_areas[loop.id] = abs(scene_polygon_area(points))
        if not loop_polygons:
            raise ValueError("No valid closed sketch loops found for constrained meshing.")
        outer_loop_id = max(loop_polygons, key=lambda key: loop_areas.get(key, 0.0))
        outer_polygon = loop_polygons[outer_loop_id]
        hole_loop_ids: list[str] = []
        for loop_id, polygon in loop_polygons.items():
            if loop_id == outer_loop_id:
                continue
            centroid = polygon_centroid(polygon)
            if self._point_in_polygon(centroid, outer_polygon):
                hole_loop_ids.append(loop_id)

        point_index_by_id: dict[str, int] = {}
        points: list[tuple[float, float]] = []

        def ensure_point(point_id: str, xy: tuple[float, float]) -> int:
            existing = point_index_by_id.get(point_id)
            if existing is not None:
                return existing
            idx = len(points)
            points.append((float(xy[0]), float(xy[1])))
            point_index_by_id[point_id] = idx
            return idx

        def loop_chain_ids(loop_id: str, polygon: list[tuple[float, float]]) -> list[str]:
            vertex_ids = [self._scene_geometry_point_id(loop_id, idx) for idx in range(len(polygon))]
            chain: list[str] = []
            for idx, vertex_id in enumerate(vertex_ids):
                chain.append(vertex_id)
                edge_id = self._scene_geometry_edge_id(loop_id, idx)
                edge_points = [
                    point
                    for point in self._scene_project.geometry_points.values()
                    if point.owner_edge_id == edge_id and point.source == "manual" and point.role == "edge_point"
                ]
                edge_points.sort(key=lambda item: float(item.param if item.param is not None else 0.0))
                for item in edge_points:
                    chain.append(item.id)
            unique_chain: list[str] = []
            for point_id in chain:
                if not unique_chain or unique_chain[-1] != point_id:
                    unique_chain.append(point_id)
            return unique_chain

        outer_chain_ids = loop_chain_ids(outer_loop_id, outer_polygon)
        outer_point_ids: list[int] = []
        for idx, point_id in enumerate(outer_chain_ids):
            if point_id not in self._scene_project.geometry_points:
                px, py = outer_polygon[idx % len(outer_polygon)]
                self._scene_project.geometry_points[point_id] = GeometryPointDef(
                    id=point_id,
                    name=f"P{idx + 1}",
                    x=float(px),
                    y=float(py),
                    owner_loop_id=outer_loop_id,
                    source="sketch",
                    role="vertex",
                )
            point = self._scene_project.geometry_points[point_id]
            outer_point_ids.append(ensure_point(point_id, (float(point.x), float(point.y))))

        segments: list[tuple[int, int]] = []

        def append_loop_segments(loop_point_ids: list[int]) -> None:
            if len(loop_point_ids) < 3:
                return
            for idx in range(len(loop_point_ids)):
                a = loop_point_ids[idx]
                b = loop_point_ids[(idx + 1) % len(loop_point_ids)]
                if a != b:
                    segments.append((a, b))

        append_loop_segments(outer_point_ids)
        hole_loops: list[BoundaryLoop] = []
        for hole_loop_id in hole_loop_ids:
            polygon = loop_polygons[hole_loop_id]
            hole_chain_ids = loop_chain_ids(hole_loop_id, polygon)
            hole_point_ids: list[int] = []
            for idx, point_id in enumerate(hole_chain_ids):
                if point_id not in self._scene_project.geometry_points:
                    px, py = polygon[idx % len(polygon)]
                    self._scene_project.geometry_points[point_id] = GeometryPointDef(
                        id=point_id,
                        name=f"H{idx + 1}",
                        x=float(px),
                        y=float(py),
                        owner_loop_id=hole_loop_id,
                        source="sketch",
                        role="vertex",
                    )
                point = self._scene_project.geometry_points[point_id]
                hole_point_ids.append(ensure_point(point_id, (float(point.x), float(point.y))))
            append_loop_segments(hole_point_ids)
            hole_loops.append(BoundaryLoop(point_ids=hole_point_ids))

        embedded_point_ids: list[int] = []
        for point in self._scene_project.geometry_points.values():
            if point.role not in {"edge_point", "interior_point"}:
                continue
            point_xy = (float(point.x), float(point.y))
            if point.role == "interior_point":
                if not self._point_in_polygon(point_xy, outer_polygon):
                    continue
                if any(self._point_in_polygon(point_xy, loop_polygons[loop_id]) for loop_id in hole_loop_ids):
                    continue
            if point.id in point_index_by_id:
                embedded_point_ids.append(point_index_by_id[point.id])
                continue
            point_index = ensure_point(point.id, point_xy)
            embedded_point_ids.append(point_index)

        region_markers = []
        for region in self._scene_project.regions.values():
            if region.source != "sketch":
                continue
            centroid = region.centroid
            if not self._point_in_polygon((float(centroid[0]), float(centroid[1])), outer_polygon):
                continue
            if any(self._point_in_polygon((float(centroid[0]), float(centroid[1])), loop_polygons[loop_id]) for loop_id in hole_loop_ids):
                continue
            mid = int(region.material_id) if region.material_id is not None and int(region.material_id) > 0 else int(self._material_assign_id_spin.value())
            region_markers.append((float(centroid[0]), float(centroid[1]), mid))
        if not region_markers:
            region_markers.append((float(polygon_centroid(outer_polygon)[0]), float(polygon_centroid(outer_polygon)[1]), int(self._material_assign_id_spin.value())))

        cdt_input = ConstrainedDelaunayInput(
            points=points,
            segments=segments,
            outer_loop=BoundaryLoop(point_ids=outer_point_ids),
            holes=hole_loops,
            regions=[RegionMarker(x=item[0], y=item[1], material_id=item[2]) for item in region_markers],
            embedded_point_ids=sorted(set(embedded_point_ids)),
        )
        result = generate_constrained_delaunay_t3(cdt_input)

        existing_material_by_id = {}
        if self._model is not None:
            existing_material_by_id = {int(material.id): material for material in self._model.materials}
        material_ids = sorted({int(element.material_id) for element in result.mesh.elements if int(element.material_id) > 0})
        materials: list[Material] = []
        for material_id in material_ids:
            existing = existing_material_by_id.get(int(material_id))
            if existing is not None:
                materials.append(
                    Material(
                        id=int(existing.id),
                        young_modulus=float(existing.young_modulus),
                        poisson_ratio=float(existing.poisson_ratio),
                        plane_stress=bool(existing.plane_stress),
                    )
                )
            else:
                materials.append(
                    Material(
                        id=int(material_id),
                        young_modulus=float(self._material_e_spin.value()),
                        poisson_ratio=float(self._material_nu_spin.value()),
                        plane_stress=bool(self._material_plane_stress.isChecked()),
                    )
                )
        if not materials:
            materials = [
                Material(
                    id=int(self._material_assign_id_spin.value()),
                    young_modulus=float(self._material_e_spin.value()),
                    poisson_ratio=float(self._material_nu_spin.value()),
                    plane_stress=bool(self._material_plane_stress.isChecked()),
                )
            ]
            default_mid = int(materials[0].id)
            for element in result.mesh.elements:
                element.material_id = default_mid
        return Model(mesh=result.mesh, materials=materials)

    def _build_part_from_sketch(self) -> None:
        if len(self._part_sketch_points) < 3:
            self._show_error(self._ui("部件生成失败", "Part build failed"), self._ui("至少需要 3 个草图点。", "At least 3 sketch points are required."))
            return
        points = list(self._part_sketch_points)
        use_constrained = self._requires_constrained_geometry_mesh()
        try:
            if use_constrained:
                model = self._build_model_from_scene_constrained()
            else:
                model = self._build_model_from_polygon_points(points)
        except Exception as exc:
            self._show_error(self._ui("部件生成失败", "Part build failed"), str(exc))
            return
        self._clear_mapping_state()
        self._set_model(model)
        mapped = self._apply_geometry_material_rules_to_mesh()
        if mapped > 0:
            self._set_model(self._model)
        self._capture_sketch_from_canvas = False
        self._part_capture_from_canvas.setChecked(False)
        self._part_sketch_history.clear()
        self._part_sketch_curve_hint = None
        self._part_sketch_undo_stack.clear()
        self._part_sketch_redo_stack.clear()
        self._reset_part_sketch_selection()
        self._log(
            self._ui(
                f"草图已生成部件（{'Constrained Delaunay' if use_constrained else 'GitHub Ear-Clipping'}）：节点 {len(model.mesh.nodes)}，单元 {len(model.mesh.elements)}，材料映射 {mapped} 个单元。",
                f"Part created from sketch with {'constrained Delaunay' if use_constrained else 'GitHub ear-clipping'}: {len(model.mesh.nodes)} nodes, {len(model.mesh.elements)} elements, mapped {mapped} elements.",
            )
        )

    @staticmethod
    def _polygon_area(points: list[tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.0
        area_twice = 0.0
        count = len(points)
        for i in range(count):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % count]
            area_twice += x0 * y1 - x1 * y0
        return 0.5 * area_twice

    @staticmethod
    def _normalize_polygon_points(points_raw: list[tuple[float, float]]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for item in points_raw:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            x = float(item[0])
            y = float(item[1])
            if points and abs(points[-1][0] - x) <= 1e-9 and abs(points[-1][1] - y) <= 1e-9:
                continue
            points.append((x, y))
        if len(points) >= 2:
            first = points[0]
            last = points[-1]
            if abs(first[0] - last[0]) <= 1e-9 and abs(first[1] - last[1]) <= 1e-9:
                points = points[:-1]
        if len(points) < 3:
            return []
        if abs(MainWindow._polygon_area(points)) <= 1e-12:
            return []
        return points

    @staticmethod
    def _polygon_from_curve_hint(hint: dict[str, object]) -> list[tuple[float, float]]:
        kind = str(hint.get("kind", "")).lower()
        bbox = hint.get("bbox")
        if kind not in {"circle", "ellipse"} or not isinstance(bbox, tuple) or len(bbox) != 4:
            return []
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        rx = 0.5 * abs(x1 - x0)
        ry = 0.5 * abs(y1 - y0)
        if rx <= 1e-12 or ry <= 1e-12:
            return []
        segments = 48
        polygon: list[tuple[float, float]] = []
        for i in range(segments):
            t = 2.0 * 3.141592653589793 * (float(i) / float(segments))
            polygon.append((cx + rx * cos(t), cy + ry * sin(t)))
        return polygon

    @staticmethod
    def _point_on_segment(point: tuple[float, float], p0: tuple[float, float], p1: tuple[float, float], tol: float = 1e-8) -> bool:
        x, y = point
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= tol * tol:
            return abs(x - x0) <= tol and abs(y - y0) <= tol
        cross = (x - x0) * dy - (y - y0) * dx
        if abs(cross) > tol * max(1.0, sqrt(seg_len2)):
            return False
        dot = (x - x0) * dx + (y - y0) * dy
        if dot < -tol:
            return False
        if dot > seg_len2 + tol:
            return False
        return True

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        if len(polygon) < 3:
            return False
        x, y = point
        inside = False
        count = len(polygon)
        for i in range(count):
            x0, y0 = polygon[i]
            x1, y1 = polygon[(i + 1) % count]
            if MainWindow._point_on_segment((x, y), (x0, y0), (x1, y1)):
                return True
            cond = (y0 > y) != (y1 > y)
            if not cond:
                continue
            t = (y - y0) / (y1 - y0)
            xin = x0 + t * (x1 - x0)
            if xin >= x:
                inside = not inside
        return inside

    @staticmethod
    def _points_close(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-8) -> bool:
        return abs(float(a[0]) - float(b[0])) <= tol and abs(float(a[1]) - float(b[1])) <= tol

    @staticmethod
    def _segment_parameter(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if abs(dx) >= abs(dy):
            if abs(dx) <= 1e-12:
                return 0.0
            return (float(point[0]) - float(start[0])) / dx
        if abs(dy) <= 1e-12:
            return 0.0
        return (float(point[1]) - float(start[1])) / dy

    @staticmethod
    def _interpolate_segment(start: tuple[float, float], end: tuple[float, float], t: float) -> tuple[float, float]:
        return (
            float(start[0]) + (float(end[0]) - float(start[0])) * float(t),
            float(start[1]) + (float(end[1]) - float(start[1])) * float(t),
        )

    @staticmethod
    def _segment_intersections(
        p0: tuple[float, float],
        p1: tuple[float, float],
        q0: tuple[float, float],
        q1: tuple[float, float],
        *,
        tol: float = 1e-9,
    ) -> list[tuple[tuple[float, float], float, float]]:
        px, py = float(p0[0]), float(p0[1])
        rx, ry = float(p1[0]) - px, float(p1[1]) - py
        qx, qy = float(q0[0]), float(q0[1])
        sx, sy = float(q1[0]) - qx, float(q1[1]) - qy

        def cross(ax: float, ay: float, bx: float, by: float) -> float:
            return ax * by - ay * bx

        results: list[tuple[tuple[float, float], float, float]] = []
        rxs = cross(rx, ry, sx, sy)
        qmpx = qx - px
        qmpy = qy - py
        qmpxr = cross(qmpx, qmpy, rx, ry)

        if abs(rxs) <= tol and abs(qmpxr) <= tol:
            candidates = [
                (p0, 0.0, MainWindow._segment_parameter(p0, q0, q1)),
                (p1, 1.0, MainWindow._segment_parameter(p1, q0, q1)),
                (q0, MainWindow._segment_parameter(q0, p0, p1), 0.0),
                (q1, MainWindow._segment_parameter(q1, p0, p1), 1.0),
            ]
            for point, tp, tq in candidates:
                if not MainWindow._point_on_segment(point, p0, p1, tol=1e-7):
                    continue
                if not MainWindow._point_on_segment(point, q0, q1, tol=1e-7):
                    continue
                duplicate = False
                for existing_point, _, _ in results:
                    if MainWindow._points_close(existing_point, point, tol=1e-7):
                        duplicate = True
                        break
                if not duplicate:
                    results.append((point, float(tp), float(tq)))
            return results

        if abs(rxs) <= tol:
            return []

        t = cross(qmpx, qmpy, sx, sy) / rxs
        u = cross(qmpx, qmpy, rx, ry) / rxs
        if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
            point = (
                px + rx * max(0.0, min(1.0, t)),
                py + ry * max(0.0, min(1.0, t)),
            )
            return [(point, float(t), float(u))]
        return []

    @staticmethod
    def _vertex_key(point: tuple[float, float], tol: float = 1e-8) -> tuple[int, int]:
        return (int(round(float(point[0]) / tol)), int(round(float(point[1]) / tol)))

    def _collect_raw_sketch_face_regions(self) -> list[dict[str, object]]:
        regions: list[dict[str, object]] = []

        active = self._normalize_polygon_points(self._part_sketch_points)
        if active:
            regions.append(
                {
                    "key": "active",
                    "points": active,
                    "label": self._ui("当前草图", "Current sketch"),
                    "area": abs(self._polygon_area(active)),
                }
            )

        for idx, item in enumerate(self._part_sketch_history):
            points_raw = item.get("points", [])
            points = self._normalize_polygon_points(points_raw if isinstance(points_raw, list) else [])
            if not points:
                hint = item.get("hint")
                if isinstance(hint, dict):
                    points = self._polygon_from_curve_hint(hint)
                    points = self._normalize_polygon_points(points)
            if not points:
                continue
            regions.append(
                {
                    "key": f"history:{idx}",
                    "points": points,
                    "label": self._ui(f"历史图形 #{idx + 1}", f"History shape #{idx + 1}"),
                    "area": abs(self._polygon_area(points)),
                }
            )
        return regions

    def _build_atomic_sketch_face_regions(self, raw_regions: list[dict[str, object]]) -> list[dict[str, object]]:
        normalized_regions: list[dict[str, object]] = []
        for item in raw_regions:
            key = str(item.get("key", "")).strip()
            label = str(item.get("label", key)).strip() or key
            points_raw = item.get("points", [])
            points = self._normalize_polygon_points(points_raw if isinstance(points_raw, list) else [])
            if not key or len(points) < 3:
                continue
            normalized_regions.append(
                {
                    "key": key,
                    "label": label,
                    "points": points,
                    "area": abs(scene_polygon_area(points)),
                }
            )

        if not normalized_regions:
            return []

        if len(normalized_regions) == 1:
            return normalized_regions

        edges: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        for region in normalized_regions:
            points = list(region["points"])
            for idx in range(len(points)):
                p0 = points[idx]
                p1 = points[(idx + 1) % len(points)]
                if self._points_close(p0, p1):
                    continue
                edges.append((str(region["key"]), p0, p1))

        if not edges:
            return normalized_regions

        split_params: list[list[float]] = [[0.0, 1.0] for _ in edges]
        for idx in range(len(edges)):
            _, p0, p1 = edges[idx]
            for jdx in range(idx + 1, len(edges)):
                _, q0, q1 = edges[jdx]
                intersections = self._segment_intersections(p0, p1, q0, q1)
                if not intersections:
                    continue
                for _, t_p, t_q in intersections:
                    split_params[idx].append(max(0.0, min(1.0, float(t_p))))
                    split_params[jdx].append(max(0.0, min(1.0, float(t_q))))

        vertices: list[tuple[float, float]] = []
        vertex_ids: dict[tuple[int, int], int] = {}
        undirected_edges: dict[tuple[int, int], None] = {}

        def ensure_vertex(point: tuple[float, float]) -> int:
            key = self._vertex_key(point)
            existing = vertex_ids.get(key)
            if existing is not None:
                return existing
            idx = len(vertices)
            vertices.append((float(point[0]), float(point[1])))
            vertex_ids[key] = idx
            return idx

        for idx, (_, p0, p1) in enumerate(edges):
            params = sorted({round(float(value), 10) for value in split_params[idx]})
            for start_t, end_t in zip(params, params[1:]):
                if float(end_t) - float(start_t) <= 1e-8:
                    continue
                start_point = self._interpolate_segment(p0, p1, float(start_t))
                end_point = self._interpolate_segment(p0, p1, float(end_t))
                if self._points_close(start_point, end_point):
                    continue
                a = ensure_vertex(start_point)
                b = ensure_vertex(end_point)
                if a == b:
                    continue
                edge_key = (a, b) if a < b else (b, a)
                undirected_edges[edge_key] = None

        if not undirected_edges:
            return normalized_regions

        adjacency: dict[int, list[tuple[int, float]]] = {}
        for a, b in undirected_edges:
            ax, ay = vertices[a]
            bx, by = vertices[b]
            adjacency.setdefault(a, []).append((b, atan2(by - ay, bx - ax)))
            adjacency.setdefault(b, []).append((a, atan2(ay - by, ax - bx)))
        for vertex_id in adjacency:
            adjacency[vertex_id].sort(key=lambda item: item[1])

        directed_edges = {
            (a, b)
            for a, b in undirected_edges
        } | {
            (b, a)
            for a, b in undirected_edges
        }
        visited_directed: set[tuple[int, int]] = set()
        raw_points_by_key = {
            str(item["key"]): [(float(x), float(y)) for x, y in item["points"]]
            for item in normalized_regions
        }
        raw_label_by_key = {str(item["key"]): str(item["label"]) for item in normalized_regions}
        raw_area_by_key = {str(item["key"]): float(item["area"]) for item in normalized_regions}
        temp_regions: list[dict[str, object]] = []
        tau = 2.0 * 3.141592653589793

        def next_clockwise_neighbor(prev_vertex: int, current_vertex: int) -> int | None:
            outgoing = adjacency.get(current_vertex, [])
            if not outgoing:
                return None
            cx, cy = vertices[current_vertex]
            px, py = vertices[prev_vertex]
            back_angle = atan2(py - cy, px - cx)
            best_neighbor: int | None = None
            best_delta = float("inf")
            for neighbor, angle in outgoing:
                delta = (back_angle - angle) % tau
                if delta <= 1e-9:
                    delta = tau
                if delta < best_delta:
                    best_delta = delta
                    best_neighbor = neighbor
            return best_neighbor

        for start_edge in sorted(directed_edges):
            if start_edge in visited_directed:
                continue
            face_vertices: list[int] = []
            current_edge = start_edge
            guard = 0
            while True:
                if current_edge in visited_directed and current_edge != start_edge:
                    face_vertices = []
                    break
                visited_directed.add(current_edge)
                from_vertex, to_vertex = current_edge
                face_vertices.append(from_vertex)
                next_vertex = next_clockwise_neighbor(from_vertex, to_vertex)
                if next_vertex is None:
                    face_vertices = []
                    break
                current_edge = (to_vertex, next_vertex)
                guard += 1
                if current_edge == start_edge:
                    break
                if guard > len(directed_edges) + 4:
                    face_vertices = []
                    break

            if len(face_vertices) < 3:
                continue

            polygon = [vertices[vertex_id] for vertex_id in face_vertices]
            normalized_polygon = self._normalize_polygon_points(polygon)
            if len(normalized_polygon) < 3:
                continue
            area = scene_polygon_area(normalized_polygon)
            if area <= 1e-8:
                continue
            centroid = polygon_centroid(normalized_polygon)
            signature_keys = [
                key
                for key, points in raw_points_by_key.items()
                if self._point_in_polygon(centroid, points)
            ]
            if not signature_keys:
                continue
            temp_regions.append(
                {
                    "signature_keys": tuple(sorted(signature_keys)),
                    "points": normalized_polygon,
                    "area": abs(area),
                    "centroid": centroid,
                }
            )

        if not temp_regions:
            return normalized_regions

        temp_regions.sort(
            key=lambda item: (
                len(item["signature_keys"]),
                str(item["signature_keys"]),
                -float(item["area"]),
                float(item["centroid"][0]),
                float(item["centroid"][1]),
            )
        )

        signature_counts: dict[tuple[str, ...], int] = {}
        atomic_regions: list[dict[str, object]] = []
        for item in temp_regions:
            signature = tuple(str(key) for key in item["signature_keys"])
            area = float(item["area"])
            centroid = item["centroid"]
            signature_counts[signature] = signature_counts.get(signature, 0) + 1
            signature_index = signature_counts[signature]

            if len(signature) == 1:
                raw_key = signature[0]
                raw_area = raw_area_by_key.get(raw_key, area)
                if abs(area - raw_area) <= max(1e-6, 1e-5 * max(abs(raw_area), 1.0)):
                    region_key = raw_key
                    label = raw_label_by_key.get(raw_key, raw_key)
                else:
                    region_key = (
                        f"split:{raw_key}:{signature_index}:"
                        f"{round(float(centroid[0]), 6)}:{round(float(centroid[1]), 6)}"
                    )
                    label = self._ui(
                        f"{raw_label_by_key.get(raw_key, raw_key)} 子区域 #{signature_index}",
                        f"{raw_label_by_key.get(raw_key, raw_key)} sub-region #{signature_index}",
                    )
            else:
                joined_labels = " ∩ ".join(raw_label_by_key.get(key, key) for key in signature)
                region_key = (
                    f"overlap:{'-'.join(signature)}:{signature_index}:"
                    f"{round(float(centroid[0]), 6)}:{round(float(centroid[1]), 6)}"
                )
                label = self._ui(
                    f"交叠区域 #{signature_index}: {joined_labels}",
                    f"Overlap region #{signature_index}: {joined_labels}",
                )

            atomic_regions.append(
                {
                    "key": region_key,
                    "points": [(float(x), float(y)) for x, y in item["points"]],
                    "label": label,
                    "area": area,
                }
            )

        if atomic_regions:
            return atomic_regions
        return normalized_regions

    @staticmethod
    def _scene_loop_id_from_key(key: str) -> str:
        return f"loop:{str(key)}"

    @staticmethod
    def _scene_region_id_from_key(key: str) -> str:
        return f"region:{str(key)}"

    @staticmethod
    def _scene_geometry_point_id(loop_id: str, index: int) -> str:
        return f"{loop_id}:pt:{int(index)}"

    @staticmethod
    def _scene_geometry_edge_id(loop_id: str, index: int) -> str:
        return f"{loop_id}:edge:{int(index)}"

    @staticmethod
    def _scene_component_color(component_id: str) -> str:
        palette = (
            "#2563eb",
            "#f59e0b",
            "#16a34a",
            "#dc2626",
            "#7c3aed",
            "#0ea5e9",
            "#ea580c",
            "#65a30d",
        )
        idx = abs(hash(component_id)) % len(palette)
        return palette[idx]

    def _reset_scene_project(self) -> None:
        self._scene_project = SceneProject()
        if hasattr(self, "_mesh_backend_combo"):
            self._scene_project.mesh_backend = str(self._mesh_backend_combo.currentData() or "builtin")
        if hasattr(self, "_mesh_global_seed_spin"):
            self._scene_project.global_mesh_size = float(self._mesh_global_seed_spin.value())
        default_component = self._scene_project.ensure_default_component()
        default_component.display_color = self._scene_component_color(default_component.id)
        self._scene_selection = SceneSelection()
        self._scene_mesh_state = SceneMeshState()
        self._last_mesh_quality_report = None
        self._mesh_bad_elements_visible = False
        self._resolved_scene_loads_by_def_id.clear()
        self._resolved_scene_bcs_by_def_id.clear()
        self._geometry_pick_point_cache.clear()
        self._geometry_pick_edge_cache.clear()
        self._sketch_face_pick_cache.clear()
        self._mesh_canvas.set_visible_elements(None)
        self._mesh_canvas.set_component_sketch_faces([])
        if hasattr(self, "_mesh_canvas"):
            self._mesh_canvas.set_load_bc_geometry_overlay([], [], set(), set(), [])
        self._refresh_component_manager()
        self._refresh_load_bc_target_set_combo()

    def _seed_scene_from_imported_model(self, *, gmsh_import: GmshT3ImportResult | None = None) -> None:
        self._reset_scene_project()
        if self._model is None:
            return

        default_component = self._scene_project.ensure_default_component()
        default_component.name = self._ui("导入组件", "Imported Component")
        default_component.display_color = self._scene_component_color(default_component.id)

        region_count = 0
        if gmsh_import is not None and gmsh_import.triangle_physical_tag_by_element_id:
            tag_to_element_ids: dict[int, list[int]] = {}
            for element in self._model.mesh.elements:
                tag = gmsh_import.triangle_physical_tag_by_element_id.get(int(element.id))
                if tag is None:
                    continue
                tag_to_element_ids.setdefault(int(tag), []).append(int(element.id))

            for idx, tag in enumerate(sorted(tag_to_element_ids)):
                region_name = gmsh_import.physical_name_by_dim_tag.get((2, tag), f"Physical-{tag}")
                if idx == 0:
                    component = default_component
                    component.name = region_name
                else:
                    component = self._scene_project.ensure_component(
                        component_id=f"component:import:{tag}",
                        name=region_name,
                    )
                component.display_color = self._scene_component_color(component.id)
                self._scene_project.regions[f"region:import:{tag}"] = RegionDef(
                    id=f"region:import:{tag}",
                    name=region_name,
                    loop_ids=[],
                    component_id=component.id,
                    material_id=int(tag),
                    source="import",
                )
                region_count += 1

        if region_count <= 0:
            material_ids = sorted({int(element.material_id) for element in self._model.mesh.elements if int(element.material_id) > 0})
            fallback_material_id = material_ids[0] if len(material_ids) == 1 else None
            self._scene_project.regions["region:import:all"] = RegionDef(
                id="region:import:all",
                name=self._ui("导入区域", "Imported Region"),
                loop_ids=[],
                component_id=default_component.id,
                material_id=fallback_material_id,
                source="import",
            )

        self._scene_project.rebuild_component_regions()
        self._sync_legacy_material_rules_from_scene()
        self._sync_component_sketch_faces_to_canvas()
        self._rebuild_geometry_pick_cache()
        self._rebuild_sketch_face_pick_cache()
        self._rebuild_scene_mesh_state()
        self._refresh_component_manager()

    def _scene_loop_points(self, loop_id: str) -> list[tuple[float, float]]:
        loop = self._scene_project.loops.get(str(loop_id))
        if loop is None:
            return []
        return [(float(x), float(y)) for x, y in loop.points]

    def _scene_region_points(self, region: RegionDef) -> list[tuple[float, float]]:
        if not region.loop_ids:
            return []
        return self._scene_loop_points(region.loop_ids[0])

    def _sync_scene_from_sketch_geometry(self) -> None:
        self._scene_project.ensure_default_component()
        sketch_faces = self._build_atomic_sketch_face_regions(self._collect_raw_sketch_face_regions())
        old_regions = {
            region_id: copy.deepcopy(region)
            for region_id, region in self._scene_project.regions.items()
            if region.source == "sketch"
        }
        self._scene_project.clear_sketch_entities()

        for item in sketch_faces:
            key = str(item.get("key", ""))
            points_raw = item.get("points", [])
            points = self._normalize_polygon_points(points_raw if isinstance(points_raw, list) else [])
            if not key or len(points) < 3:
                continue
            loop_id = self._scene_loop_id_from_key(key)
            region_id = self._scene_region_id_from_key(key)
            old_region = old_regions.get(region_id)
            label = str(item.get("label", key))
            loop = LoopDef(
                id=loop_id,
                name=label,
                points=[(float(x), float(y)) for x, y in points],
                is_closed=True,
                is_valid=True,
                source="sketch",
            )
            self._scene_project.loops[loop_id] = loop
            point_ids: list[str] = []
            for idx, (px, py) in enumerate(points):
                point_id = self._scene_geometry_point_id(loop_id, idx)
                point_ids.append(point_id)
                self._scene_project.geometry_points[point_id] = GeometryPointDef(
                    id=point_id,
                    name=f"P{idx + 1}",
                    x=float(px),
                    y=float(py),
                    owner_loop_id=loop_id,
                    owner_edge_id=None,
                    param=float(idx),
                    source="sketch",
                    role="vertex",
                )
            for idx in range(len(point_ids)):
                start_id = point_ids[idx]
                end_id = point_ids[(idx + 1) % len(point_ids)]
                edge_id = self._scene_geometry_edge_id(loop_id, idx)
                self._scene_project.geometry_edges[edge_id] = GeometryEdgeDef(
                    id=edge_id,
                    name=f"E{idx + 1}",
                    loop_id=loop_id,
                    start_point_id=start_id,
                    end_point_id=end_id,
                    source="sketch",
                )

            material_id: int | None = None
            if old_region is not None:
                material_id = old_region.material_id
            if material_id is None:
                legacy = self._material_geometry_region_rules.get(key)
                if legacy is None:
                    legacy = self._material_geometry_region_rules.get(region_id.replace("region:", "", 1))
                if isinstance(legacy, dict):
                    legacy_mid = int(legacy.get("material_id", 0))
                    material_id = legacy_mid if legacy_mid > 0 else None

            component_id = old_region.component_id if old_region is not None else self._scene_project.default_component_id
            if component_id not in self._scene_project.components:
                component_id = self._scene_project.default_component_id

            region = RegionDef(
                id=region_id,
                name=label,
                loop_ids=[loop_id],
                hole_loop_ids=[],
                component_id=component_id,
                material_id=material_id,
                mesh_size=old_region.mesh_size if old_region is not None else None,
                display_color=old_region.display_color if old_region is not None else "#93c5fd",
                visible=old_region.visible if old_region is not None else True,
                area=abs(scene_polygon_area(points)),
                centroid=polygon_centroid(points),
                source="sketch",
            )
            self._scene_project.regions[region_id] = region

        self._scene_project.rebuild_component_regions()
        self._sync_legacy_material_rules_from_scene()
        self._sync_component_sketch_faces_to_canvas()
        self._rebuild_geometry_pick_cache()
        self._rebuild_sketch_face_pick_cache()
        self._sync_pickable_sketch_faces_to_canvas()
        self._refresh_component_manager()
        self._rebuild_scene_mesh_state()
        if (
            self._scene_project.geometry_points
            and hasattr(self, "_load_bc_target_mode_combo")
            and (self._model is None or not self._model.mesh.nodes)
        ):
            idx = self._load_bc_target_mode_combo.findData("geometry")
            if idx >= 0 and self._load_bc_target_mode_combo.currentData() != "geometry":
                self._load_bc_target_mode_combo.setCurrentIndex(idx)

    def _sync_legacy_material_rules_from_scene(self) -> None:
        rules: dict[str, dict[str, object]] = {}
        for region in self._scene_project.regions.values():
            if region.material_id is None or int(region.material_id) <= 0:
                continue
            points = self._scene_region_points(region)
            points = self._normalize_polygon_points(points)
            if len(points) < 3:
                continue
            key = region.id.replace("region:", "", 1)
            rules[key] = {
                "material_id": int(region.material_id),
                "points": [(float(x), float(y)) for x, y in points],
                "label": str(region.name),
            }
        self._material_geometry_region_rules = rules

    @staticmethod
    def _scene_loop_edge_ids(loop_id: str, count: int) -> list[str]:
        return [f"{loop_id}:edge:{int(idx)}" for idx in range(max(int(count), 0))]

    @staticmethod
    def _scene_loop_point_ids(loop_id: str, count: int) -> list[str]:
        return [f"{loop_id}:pt:{int(idx)}" for idx in range(max(int(count), 0))]

    def _mesh_backend_name(self) -> str:
        if hasattr(self, "_mesh_backend_combo"):
            data = self._mesh_backend_combo.currentData()
            if isinstance(data, str) and data.strip():
                return data
        return str(getattr(self._scene_project, "mesh_backend", "builtin"))

    def _mesh_regions_from_scene(self) -> list[MeshRegion]:
        regions: list[MeshRegion] = []
        default_material_id = int(self._material_assign_id_spin.value()) if hasattr(self, "_material_assign_id_spin") else 1
        for region in self._scene_project.regions.values():
            points = self._normalize_polygon_points(self._scene_region_points(region))
            if len(points) < 3:
                continue
            loop_id = region.loop_ids[0] if region.loop_ids else ""
            hole_points = [
                self._normalize_polygon_points(self._scene_loop_points(hole_loop_id))
                for hole_loop_id in region.hole_loop_ids
            ]
            edge_ids = self._scene_loop_edge_ids(loop_id, len(points)) if loop_id else []
            point_ids = self._scene_loop_point_ids(loop_id, len(points)) if loop_id else []
            regions.append(
                MeshRegion(
                    id=str(region.id),
                    name=str(region.name),
                    points=points,
                    material_id=int(region.material_id) if region.material_id is not None and int(region.material_id) > 0 else default_material_id,
                    mesh_size=float(region.mesh_size) if region.mesh_size is not None and float(region.mesh_size) > 0.0 else None,
                    hole_points=[points_item for points_item in hole_points if len(points_item) >= 3],
                    edge_ids=edge_ids,
                    point_ids=point_ids,
                    component_id=str(region.component_id),
                )
            )
        if regions:
            return regions

        points = self._normalize_polygon_points(list(self._part_sketch_points))
        if len(points) >= 3:
            regions.append(
                MeshRegion(
                    id="region:active",
                    name="Active Sketch Region",
                    points=points,
                    material_id=default_material_id,
                    mesh_size=None,
                    edge_ids=self._scene_loop_edge_ids("loop:active", len(points)),
                    point_ids=self._scene_loop_point_ids("loop:active", len(points)),
                    component_id=self._scene_project.default_component_id,
                )
            )
        return regions

    def _mesh_geometry_points_from_scene(self) -> dict[str, MeshGeometryPoint]:
        points: dict[str, MeshGeometryPoint] = {}
        for point_id, point in self._scene_project.geometry_points.items():
            points[str(point_id)] = MeshGeometryPoint(
                id=str(point_id),
                x=float(point.x),
                y=float(point.y),
                owner_edge_id=str(point.owner_edge_id) if point.owner_edge_id else None,
                role=point.role,
            )
        for region in self._mesh_regions_from_scene():
            for idx, point_id in enumerate(region.point_ids):
                if idx >= len(region.points) or point_id in points:
                    continue
                x, y = region.points[idx]
                points[str(point_id)] = MeshGeometryPoint(id=str(point_id), x=float(x), y=float(y), role="vertex")
        return points

    def _mesh_geometry_edges_from_scene(self) -> dict[str, MeshGeometryEdge]:
        edges: dict[str, MeshGeometryEdge] = {}
        for edge_id, edge in self._scene_project.geometry_edges.items():
            edges[str(edge_id)] = MeshGeometryEdge(
                id=str(edge_id),
                start_point_id=str(edge.start_point_id),
                end_point_id=str(edge.end_point_id),
            )
        for region in self._mesh_regions_from_scene():
            for idx, edge_id in enumerate(region.edge_ids):
                if edge_id in edges or idx >= len(region.point_ids):
                    continue
                edges[str(edge_id)] = MeshGeometryEdge(
                    id=str(edge_id),
                    start_point_id=str(region.point_ids[idx]),
                    end_point_id=str(region.point_ids[(idx + 1) % len(region.point_ids)]),
                )
        return edges

    def _current_mesh_seed(self) -> MeshSeed:
        global_size = float(self._mesh_global_seed_spin.value()) if hasattr(self, "_mesh_global_seed_spin") else float(self._scene_project.global_mesh_size)
        region_sizes = {
            str(region.id): float(region.mesh_size)
            for region in self._scene_project.regions.values()
            if region.mesh_size is not None and float(region.mesh_size) > 0.0
        }
        return MeshSeed(
            global_size=global_size,
            edge_seeds={str(edge_id): float(size) for edge_id, size in self._scene_project.edge_mesh_sizes.items() if float(size) > 0.0},
            region_sizes=region_sizes,
            bias={},
        )

    def _current_mesh_controls(self) -> list[MeshControl]:
        controls: list[MeshControl] = []
        algorithm_text = str(self._mesh_algo_combo.currentData()) if hasattr(self, "_mesh_algo_combo") else "delaunay"
        algorithm = "structured" if algorithm_text == "structured" else "free"
        for region in self._scene_project.regions.values():
            controls.append(
                MeshControl(
                    region_id=str(region.id),
                    algorithm=algorithm,
                    element_type="T3",
                    allow_boundary_preserve=True,
                )
            )
        if not controls:
            controls.append(MeshControl(region_id=None, algorithm=algorithm, element_type="T3", allow_boundary_preserve=True))
        return controls

    def _preserve_node_ids_for_mesh_ops(self) -> set[int]:
        preserve_ids = {
            int(bc.node_id)
            for bc in (self._model.boundary_conditions if self._model is not None else [])
            if int(bc.node_id) > 0
        }
        preserve_ids.update(
            int(load.node_id)
            for load in (self._model.loads if self._model is not None else [])
            if int(load.node_id) > 0
        )
        for node_ids in self._scene_mesh_state.set_to_node_ids.values():
            preserve_ids.update(int(node_id) for node_id in node_ids if int(node_id) > 0)
        return preserve_ids

    def _mesh_quality_highlight_ids(self) -> set[int]:
        if not self._mesh_bad_elements_visible or self._last_mesh_quality_report is None:
            return set()
        return {int(item) for item in self._last_mesh_quality_report.bad_element_ids}

    def _refresh_mesh_quality_summary(self) -> None:
        report = self._scene_mesh_state.quality_report or self._last_mesh_quality_report
        if hasattr(self, "_mesh_quality_summary"):
            if report is None:
                self._mesh_quality_summary.setText(self._ui("尚无网格质量报告。", "No mesh quality report yet."))
            else:
                self._mesh_quality_summary.setText(report.summary)
        self._mesh_canvas.set_quality_bad_elements(self._mesh_quality_highlight_ids())

    def _current_mesh_selection_element_ids(self) -> set[int]:
        if self._selected_face_region_element_ids:
            return {int(item) for item in self._selected_face_region_element_ids}
        return {int(item) for item in self._selected_element_ids}

    def _current_mesh_selection_region_ids(self) -> list[str]:
        return self._selected_scene_region_ids_for_component_assignment()

    def _update_mesh_selection_hint(self) -> None:
        if not hasattr(self, "_mesh_selection_hint"):
            return
        region_ids = self._current_mesh_selection_region_ids()
        element_ids = self._current_mesh_selection_element_ids()
        edge_count = len(self._selected_geometry_edge_ids)
        if edge_count > 0:
            self._mesh_selection_hint.setText(
                self._ui(
                    f"已选 {edge_count} 条边，可继续点选增减，Esc 结束。",
                    f"{edge_count} edge(s) selected. Keep clicking to add/remove, Esc to finish.",
                )
            )
            return
        if element_ids:
            self._mesh_selection_hint.setText(
                self._ui(
                    f"已选 {len(element_ids)} 个单元，映射 {len(region_ids)} 个区域。",
                    f"{len(element_ids)} element(s) selected, mapped to {len(region_ids)} region(s).",
                )
            )
            return
        if region_ids:
            self._mesh_selection_hint.setText(
                self._ui(
                    f"已选 {len(region_ids)} 个区域。",
                    f"{len(region_ids)} region(s) selected.",
                )
            )
            return
        self._mesh_selection_hint.setText(
            self._ui("未选择局部网格目标。", "No local mesh target selected.")
        )

    def _materials_for_mesh(self, mesh: Mesh) -> list[Material]:
        existing_by_id = {
            int(material.id): copy.deepcopy(material)
            for material in (self._model.materials if self._model is not None else [])
        }
        material_ids = sorted({int(element.material_id) for element in mesh.elements if int(element.material_id) > 0})
        if not material_ids:
            material_ids = [int(self._material_assign_id_spin.value())]
        materials: list[Material] = []
        for material_id in material_ids:
            material = existing_by_id.get(material_id)
            if material is None:
                material = Material(
                    id=int(material_id),
                    young_modulus=float(self._material_e_spin.value()),
                    poisson_ratio=float(self._material_nu_spin.value()),
                    plane_stress=bool(self._material_plane_stress.isChecked()),
                )
            material.id = int(material_id)
            materials.append(material)
        return materials

    def _apply_mesh_generation_result(self, result, *, operation_label: str) -> None:
        valid_node_ids = {int(node.id) for node in result.mesh.nodes}
        boundary_conditions = []
        loads = []
        if self._model is not None:
            boundary_conditions = [
                copy.deepcopy(item)
                for item in self._model.boundary_conditions
                if int(item.node_id) in valid_node_ids
            ]
            loads = [
                copy.deepcopy(item)
                for item in self._model.loads
                if int(item.node_id) in valid_node_ids
            ]
        model = Model(
            mesh=result.mesh,
            materials=self._materials_for_mesh(result.mesh),
            boundary_conditions=boundary_conditions,
            loads=loads,
        )
        self._set_model(model)
        self._scene_mesh_state.quality_report = result.quality_report
        self._scene_mesh_state.warnings = list(result.warnings)
        self._scene_mesh_state.timings_ms = dict(result.timings_ms)
        self._last_mesh_quality_report = result.quality_report
        if result.region_to_element_ids:
            self._scene_mesh_state.region_to_element_ids.update(
                {str(region_id): list(element_ids) for region_id, element_ids in result.region_to_element_ids.items()}
            )
        if result.edge_to_node_ids:
            self._scene_mesh_state.edge_to_node_ids.update(
                {str(edge_id): list(node_ids) for edge_id, node_ids in result.edge_to_node_ids.items()}
            )
        if result.point_to_node_ids:
            self._scene_mesh_state.point_to_node_ids.update(
                {str(point_id): list(node_ids) for point_id, node_ids in result.point_to_node_ids.items()}
            )
        self._refresh_scene_geometry_set_mesh_mappings()
        unresolved = self._resolve_scene_load_bc_definitions_to_model(rebuild_state=False)
        if unresolved:
            self._scene_mesh_state.warnings.extend(unresolved)
        self._sync_editor_tables_from_model()
        self._update_model_tree()
        self._refresh_readiness_state()
        self._refresh_mesh_quality_summary()
        self._mark_results_stale(operation_label)
        warning_text = f", warnings={len(self._scene_mesh_state.warnings)}" if self._scene_mesh_state.warnings else ""
        self._log(
            f"{operation_label}: nodes={len(result.mesh.nodes)}, elements={len(result.mesh.elements)}, "
            f"bad={result.quality_report.bad_element_count}, backend={self._mesh_backend_name()}{warning_text}."
        )

    def _run_mesh_backend_operation(
        self,
        *,
        operation: str,
        target_region_ids: list[str] | None = None,
        target_element_ids: set[int] | None = None,
    ) -> bool:
        self._sync_scene_from_sketch_geometry()
        request = MeshGenerationRequest(
            regions=self._mesh_regions_from_scene(),
            seed=self._current_mesh_seed(),
            controls=self._current_mesh_controls(),
            geometry_points=self._mesh_geometry_points_from_scene(),
            geometry_edges=self._mesh_geometry_edges_from_scene(),
            backend=self._mesh_backend_name(),
            operation=operation,
            existing_mesh=self._model.mesh if self._model is not None else None,
            target_region_ids=set(target_region_ids or []),
            target_element_ids={int(item) for item in (target_element_ids or set())},
            preserve_node_ids=self._preserve_node_ids_for_mesh_ops(),
        )
        try:
            backend = mesher_backend(request.backend)
            result = backend.generate(request)
        except Exception as exc:
            self._show_error(self._ui("网格操作失败", "Mesh operation failed"), str(exc))
            return False
        self._apply_mesh_generation_result(result, operation_label=f"Mesh {operation}")
        return True

    def _sync_component_sketch_faces_to_canvas(self) -> None:
        faces: list[dict[str, object]] = []
        for region in self._scene_project.regions.values():
            points = self._normalize_polygon_points(self._scene_region_points(region))
            if len(points) < 3:
                continue
            component = self._scene_project.components.get(region.component_id)
            if component is None or not component.visible:
                continue
            color = component.display_color or self._scene_component_color(component.id)
            faces.append({"points": points, "color": color})
        self._mesh_canvas.set_component_sketch_faces(faces)

    def _rebuild_scene_mesh_state(self) -> None:
        self._scene_mesh_state = SceneMeshState(solver_model=self._model)
        if self._model is None:
            self._mesh_canvas.set_visible_elements(None)
            return

        node_by_id = {int(node.id): node for node in self._model.mesh.nodes}
        region_polygons: dict[str, list[tuple[float, float]]] = {}
        for region in self._scene_project.regions.values():
            polygon = self._normalize_polygon_points(self._scene_region_points(region))
            if len(polygon) >= 3:
                region_polygons[region.id] = polygon

        region_to_element_ids: dict[str, list[int]] = {region_id: [] for region_id in self._scene_project.regions}

        for element in self._model.mesh.elements:
            if len(element.connectivity) != 3:
                continue
            nodes = [node_by_id.get(int(node_id)) for node_id in element.connectivity]
            if any(node is None for node in nodes):
                continue
            cx = (float(nodes[0].x) + float(nodes[1].x) + float(nodes[2].x)) / 3.0
            cy = (float(nodes[0].y) + float(nodes[1].y) + float(nodes[2].y)) / 3.0

            matched_region_id: str | None = None
            for region_id, polygon in region_polygons.items():
                if self._point_in_polygon((cx, cy), polygon):
                    matched_region_id = region_id
                    break

            if matched_region_id is None:
                for region in self._scene_project.regions.values():
                    if region.material_id is not None and int(region.material_id) == int(element.material_id):
                        matched_region_id = region.id
                        break

            if matched_region_id is None and len(self._scene_project.regions) == 1:
                matched_region_id = next(iter(self._scene_project.regions.keys()))

            if matched_region_id is not None:
                region_to_element_ids.setdefault(matched_region_id, []).append(int(element.id))

        component_to_element_ids: dict[str, list[int]] = {component_id: [] for component_id in self._scene_project.components}
        for region_id, element_ids in region_to_element_ids.items():
            region = self._scene_project.regions.get(region_id)
            if region is None:
                continue
            component_to_element_ids.setdefault(region.component_id, []).extend(element_ids)
        for component_id, element_ids in component_to_element_ids.items():
            component_to_element_ids[component_id] = sorted({int(item) for item in element_ids})

        self._scene_mesh_state.region_to_element_ids = {
            region_id: sorted({int(item) for item in element_ids})
            for region_id, element_ids in region_to_element_ids.items()
        }
        self._scene_mesh_state.component_to_element_ids = component_to_element_ids
        if self._model.mesh.nodes:
            xs = [float(node.x) for node in self._model.mesh.nodes]
            ys = [float(node.y) for node in self._model.mesh.nodes]
            extent = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        else:
            extent = 1.0
        geom_tol = max(extent * 1e-6, 1e-7)
        point_to_node_ids: dict[str, list[int]] = {}
        for point_id, point in self._scene_project.geometry_points.items():
            matched = []
            for node in self._model.mesh.nodes:
                if hypot(float(node.x) - float(point.x), float(node.y) - float(point.y)) <= geom_tol:
                    matched.append(int(node.id))
            point_to_node_ids[point_id] = sorted(set(matched))
        edge_to_node_ids: dict[str, list[int]] = {}
        for edge_id, edge in self._scene_project.geometry_edges.items():
            start = self._scene_project.geometry_points.get(edge.start_point_id)
            end = self._scene_project.geometry_points.get(edge.end_point_id)
            if start is None or end is None:
                edge_to_node_ids[edge_id] = []
                continue
            start_xy = (float(start.x), float(start.y))
            end_xy = (float(end.x), float(end.y))
            nodes_on_edge: list[tuple[float, int]] = []
            for node in self._model.mesh.nodes:
                point_xy = (float(node.x), float(node.y))
                if not self._point_on_segment(point_xy, start_xy, end_xy, tol=geom_tol * 2.0):
                    continue
                param = self._segment_parameter(point_xy, start_xy, end_xy)
                nodes_on_edge.append((float(param), int(node.id)))
            nodes_on_edge.sort(key=lambda item: item[0])
            edge_to_node_ids[edge_id] = [node_id for _, node_id in nodes_on_edge]
        self._scene_mesh_state.point_to_node_ids = point_to_node_ids
        self._scene_mesh_state.edge_to_node_ids = edge_to_node_ids
        set_to_node_ids: dict[str, list[int]] = {}
        set_to_element_ids: dict[str, list[int]] = {}
        node_id_set = {int(node.id) for node in self._model.mesh.nodes}
        for set_id, geometry_set in self._scene_project.geometry_sets.items():
            if geometry_set.entity_type == "point":
                ids = []
                if geometry_set.binding_mode == "geometry":
                    for point_id in geometry_set.entity_ids:
                        ids.extend(point_to_node_ids.get(str(point_id), []))
                else:
                    for token in geometry_set.entity_ids:
                        try:
                            node_id = int(token)
                        except (TypeError, ValueError):
                            continue
                        if node_id in node_id_set:
                            ids.append(node_id)
                set_to_node_ids[set_id] = sorted(set(ids))
            elif geometry_set.entity_type == "edge":
                ids = []
                if geometry_set.binding_mode == "geometry":
                    for edge_id in geometry_set.entity_ids:
                        ids.extend(edge_to_node_ids.get(str(edge_id), []))
                else:
                    for token in geometry_set.entity_ids:
                        try:
                            node_id = int(token)
                        except (TypeError, ValueError):
                            continue
                        if node_id in node_id_set:
                            ids.append(node_id)
                set_to_node_ids[set_id] = sorted(set(ids))
            elif geometry_set.entity_type == "region":
                elem_ids: list[int] = []
                for region_id in geometry_set.entity_ids:
                    elem_ids.extend(self._scene_mesh_state.region_to_element_ids.get(region_id, []))
                set_to_element_ids[set_id] = sorted(set(int(item) for item in elem_ids))
            elif geometry_set.entity_type == "component":
                elem_ids = []
                for component_id in geometry_set.entity_ids:
                    elem_ids.extend(component_to_element_ids.get(component_id, []))
                set_to_element_ids[set_id] = sorted(set(int(item) for item in elem_ids))
        for set_id, node_ids in set_to_node_ids.items():
            if not node_ids:
                continue
            node_id_filter = {int(item) for item in node_ids}
            elem_ids = []
            for element in self._model.mesh.elements:
                if any(int(node_id) in node_id_filter for node_id in element.connectivity):
                    elem_ids.append(int(element.id))
            if elem_ids:
                set_to_element_ids[set_id] = sorted(set(elem_ids))
        self._scene_mesh_state.set_to_node_ids = set_to_node_ids
        self._scene_mesh_state.set_to_element_ids = set_to_element_ids
        self._sync_load_bc_geometry_overlay()
        self._apply_component_visibility_filter_to_canvas()

    def _refresh_scene_geometry_set_mesh_mappings(self) -> None:
        if self._model is None:
            return
        node_id_set = {int(node.id) for node in self._model.mesh.nodes}
        set_to_node_ids: dict[str, list[int]] = {}
        set_to_element_ids: dict[str, list[int]] = {}
        for set_id, geometry_set in self._scene_project.geometry_sets.items():
            if geometry_set.entity_type == "point":
                ids: list[int] = []
                if geometry_set.binding_mode == "geometry":
                    for point_id in geometry_set.entity_ids:
                        ids.extend(self._scene_mesh_state.point_to_node_ids.get(str(point_id), []))
                else:
                    for token in geometry_set.entity_ids:
                        try:
                            node_id = int(token)
                        except (TypeError, ValueError):
                            continue
                        if node_id in node_id_set:
                            ids.append(node_id)
                set_to_node_ids[set_id] = sorted(set(int(item) for item in ids))
            elif geometry_set.entity_type == "edge":
                ids = []
                if geometry_set.binding_mode == "geometry":
                    for edge_id in geometry_set.entity_ids:
                        ids.extend(self._scene_mesh_state.edge_to_node_ids.get(str(edge_id), []))
                else:
                    for token in geometry_set.entity_ids:
                        try:
                            node_id = int(token)
                        except (TypeError, ValueError):
                            continue
                        if node_id in node_id_set:
                            ids.append(node_id)
                set_to_node_ids[set_id] = sorted(set(int(item) for item in ids))
            elif geometry_set.entity_type == "region":
                elem_ids: list[int] = []
                for region_id in geometry_set.entity_ids:
                    elem_ids.extend(self._scene_mesh_state.region_to_element_ids.get(str(region_id), []))
                set_to_element_ids[set_id] = sorted(set(int(item) for item in elem_ids))
            elif geometry_set.entity_type == "component":
                elem_ids = []
                for component_id in geometry_set.entity_ids:
                    elem_ids.extend(self._scene_mesh_state.component_to_element_ids.get(str(component_id), []))
                set_to_element_ids[set_id] = sorted(set(int(item) for item in elem_ids))

        for set_id, node_ids in set_to_node_ids.items():
            if not node_ids:
                continue
            node_id_filter = {int(item) for item in node_ids}
            elem_ids = []
            for element in self._model.mesh.elements:
                if any(int(node_id) in node_id_filter for node_id in element.connectivity):
                    elem_ids.append(int(element.id))
            if elem_ids:
                set_to_element_ids[set_id] = sorted(set(elem_ids))
        self._scene_mesh_state.set_to_node_ids = set_to_node_ids
        self._scene_mesh_state.set_to_element_ids = set_to_element_ids

    def _apply_component_visibility_filter_to_canvas(self) -> None:
        if self._model is None:
            self._mesh_canvas.set_visible_elements(None)
            return

        all_element_ids = {int(element.id) for element in self._model.mesh.elements}
        if not all_element_ids:
            self._mesh_canvas.set_visible_elements(set())
            return

        visible_component_ids = {
            component.id
            for component in self._scene_project.components.values()
            if component.visible
        }
        if not visible_component_ids:
            self._mesh_canvas.set_visible_elements(set())
            return

        visible_element_ids: set[int] = set()
        for component_id in visible_component_ids:
            visible_element_ids.update(self._scene_mesh_state.component_to_element_ids.get(component_id, []))

        if not visible_element_ids:
            if self._scene_project.regions:
                self._mesh_canvas.set_visible_elements(set())
            else:
                self._mesh_canvas.set_visible_elements(None)
            return

        scope_element_ids = self._results_scope_element_ids()
        if scope_element_ids is not None:
            visible_element_ids = set(visible_element_ids).intersection(scope_element_ids)

        if visible_element_ids == all_element_ids:
            self._mesh_canvas.set_visible_elements(None)
            return
        self._mesh_canvas.set_visible_elements(visible_element_ids)

    def _component_manager_selected_id(self) -> str | None:
        if not hasattr(self, "_component_manager_table"):
            return None
        row = self._component_manager_table.currentRow()
        if row < 0:
            return None
        item = self._component_manager_table.item(row, 0)
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        if not isinstance(data, str):
            return None
        return data

    def _refresh_component_manager(self) -> None:
        if not hasattr(self, "_component_manager_table"):
            return
        selected_component_id = self._component_manager_selected_id() or self._scene_selection.active_component_id
        components = sorted(self._scene_project.components.values(), key=lambda item: item.id)
        self._component_manager_table.blockSignals(True)
        self._component_manager_table.setRowCount(len(components))
        selected_row = -1
        for row, component in enumerate(components):
            elem_count = len(self._scene_mesh_state.component_to_element_ids.get(component.id, []))
            values = [
                component.id,
                component.name,
                str(len(component.region_ids)),
                str(elem_count),
                self._ui("显示" if component.visible else "隐藏", "Visible" if component.visible else "Hidden"),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, component.id)
                self._component_manager_table.setItem(row, col, item)
            if selected_component_id is not None and component.id == selected_component_id:
                selected_row = row
        self._component_manager_table.blockSignals(False)
        if selected_row >= 0:
            self._component_manager_table.selectRow(selected_row)
        elif components:
            self._component_manager_table.selectRow(0)
        self._sync_component_sketch_faces_to_canvas()
        self._refresh_results_scope_targets()

    def _create_component_from_selected_regions(self) -> None:
        region_ids = self._selected_scene_region_ids_for_component_assignment()
        if not region_ids:
            self._show_error(
                self._ui("组件创建失败", "Create component failed"),
                self._ui("请先在画布中选择区域（草图面或网格面）。", "Please pick at least one region first."),
            )
            return
        component_name, ok = QInputDialog.getText(
            self,
            self._ui("新建组件", "Create Component"),
            self._ui("组件名称", "Component Name"),
            text=f"Component-{len(self._scene_project.components)}",
        )
        if not ok:
            return
        component_name = component_name.strip() or f"Component-{len(self._scene_project.components)}"
        component = self._scene_project.ensure_component(name=component_name)
        component.display_color = self._scene_component_color(component.id)

        changed = 0
        for region_id in region_ids:
            region = self._scene_project.regions.get(region_id)
            if region is None:
                continue
            region.component_id = component.id
            changed += 1
        self._scene_selection.active_region_ids = set(region_ids)
        self._component_face_hint.setText(
            self._ui(
                f"已选区域: {len(region_ids)}，已归属到 {component.name}",
                f"Selected {len(region_ids)} region(s), assigned to {component.name}",
            )
        )
        self._scene_project.rebuild_component_regions()
        self._rebuild_scene_mesh_state()
        self._refresh_component_manager()
        self._log(
            self._ui(
                f"已创建组件 {component.name}，并分配 {changed} 个区域。",
                f"Created component {component.name} and assigned {changed} region(s).",
            )
        )

    def _rename_selected_component(self) -> None:
        component_id = self._component_manager_selected_id()
        if component_id is None:
            return
        component = self._scene_project.components.get(component_id)
        if component is None:
            return
        new_name, ok = QInputDialog.getText(
            self,
            self._ui("重命名组件", "Rename Component"),
            self._ui("新名称", "New Name"),
            text=component.name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        component.name = new_name
        self._refresh_component_manager()

    def _toggle_selected_component_visibility(self) -> None:
        component_id = self._component_manager_selected_id()
        if component_id is None:
            return
        component = self._scene_project.components.get(component_id)
        if component is None:
            return
        component.visible = not component.visible
        self._rebuild_scene_mesh_state()
        self._refresh_component_manager()

    def _isolate_selected_component(self) -> None:
        component_id = self._component_manager_selected_id()
        if component_id is None:
            return
        for item in self._scene_project.components.values():
            item.visible = item.id == component_id
            item.isolated = item.id == component_id
        self._scene_selection.active_component_id = component_id
        self._rebuild_scene_mesh_state()
        self._selected_element_ids = set(self._scene_mesh_state.component_to_element_ids.get(component_id, []))
        self._mesh_canvas.set_highlighted_elements(self._selected_element_ids)
        self._refresh_component_manager()

    def _show_all_components(self) -> None:
        for item in self._scene_project.components.values():
            item.visible = True
            item.isolated = False
        self._scene_selection.active_component_id = None
        self._selected_element_ids.clear()
        self._mesh_canvas.set_highlighted_elements(set())
        self._rebuild_scene_mesh_state()
        self._refresh_component_manager()

    def _on_component_manager_selection_changed(self) -> None:
        component_id = self._component_manager_selected_id()
        self._scene_selection.active_component_id = component_id
        if component_id is None:
            self._selected_element_ids.clear()
            self._mesh_canvas.set_highlighted_elements(set())
            return
        self._selected_element_ids = set(self._scene_mesh_state.component_to_element_ids.get(component_id, []))
        self._mesh_canvas.set_highlighted_elements(self._selected_element_ids)

    def _region_ids_from_element_ids(self, element_ids: list[int] | set[int]) -> list[str]:
        selected = {int(value) for value in element_ids}
        if not selected:
            return []
        scored: list[tuple[int, str]] = []
        for region_id, ids in self._scene_mesh_state.region_to_element_ids.items():
            region_set = {int(item) for item in ids}
            overlap = len(selected.intersection(region_set))
            if overlap > 0:
                scored.append((overlap, region_id))
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [region_id for _, region_id in scored]
        if len(self._scene_project.regions) == 1:
            return [next(iter(self._scene_project.regions.keys()))]
        return []

    def _selected_scene_region_ids_for_component_assignment(self) -> list[str]:
        region_ids: set[str] = set()
        for key in self._selected_face_region_sketch_keys:
            region_id = self._scene_region_id_from_key(str(key))
            if region_id in self._scene_project.regions:
                region_ids.add(region_id)
        for region_id in self._scene_selection.active_region_ids:
            if region_id in self._scene_project.regions:
                region_ids.add(region_id)
        selected_element_ids = set(self._selected_face_region_element_ids)
        if not selected_element_ids:
            selected_element_ids = set(self._selected_element_ids)
        for region_id in self._region_ids_from_element_ids(selected_element_ids):
            if region_id in self._scene_project.regions:
                region_ids.add(region_id)
        return sorted(region_ids)

    @staticmethod
    def _is_region_pick_context(context: str | None) -> bool:
        return context in {
            "material_face",
            "material_face_sketch",
            "component_face",
            "component_face_sketch",
            "mesh_region",
            "mesh_region_sketch",
            "load_bc_region",
            "load_bc_region_sketch",
        }

    def _toggle_sketch_region_selection(self, key: str) -> tuple[bool, str]:
        label = str(key)
        regions_by_key = {
            str(item["key"]): item
            for item in self._collect_sketch_face_regions()
        }
        region = regions_by_key.get(str(key))
        if region is not None:
            label = str(region.get("label", key))
        selected_keys = set(self._selected_face_region_sketch_keys)
        added = str(key) not in selected_keys
        if added:
            selected_keys.add(str(key))
        else:
            selected_keys.discard(str(key))
        self._selected_face_region_sketch_keys = selected_keys
        self._selected_face_region_element_ids.clear()
        self._selected_element_ids.clear()
        self._mesh_canvas.set_highlighted_elements(set())
        self._update_sketch_face_highlight()
        self._scene_selection.active_region_ids = {
            self._scene_region_id_from_key(item)
            for item in self._selected_face_region_sketch_keys
            if self._scene_region_id_from_key(item) in self._scene_project.regions
        }
        return added, label

    def _toggle_element_region_selection(self, region_ids: set[int]) -> tuple[bool, int, int]:
        normalized_ids = {int(item) for item in region_ids}
        if not normalized_ids:
            return False, 0, 0
        selected_ids = set(self._selected_face_region_element_ids)
        added = not normalized_ids.issubset(selected_ids)
        if added:
            selected_ids.update(normalized_ids)
        else:
            selected_ids.difference_update(normalized_ids)
        self._selected_face_region_element_ids = selected_ids
        self._selected_face_region_sketch_keys.clear()
        self._selected_element_ids = set(selected_ids)
        self._mesh_canvas.set_highlighted_elements(selected_ids)
        self._mesh_canvas.set_highlighted_sketch_faces([])
        self._scene_selection.active_region_ids = set(self._region_ids_from_element_ids(selected_ids))
        return added, len(normalized_ids), len(selected_ids)

    def _update_material_region_selection_hint(self) -> None:
        region_count = len(self._selected_scene_region_ids_for_component_assignment())
        element_count = len(self._selected_face_region_element_ids)
        sketch_count = len(self._selected_face_region_sketch_keys)
        if element_count > 0:
            self._material_face_hint.setText(
                self._ui(
                    f"已选 {element_count} 个单元，映射 {region_count} 个区域；可继续点选增减，Esc 结束。",
                    f"Selected {element_count} elements mapped to {region_count} region(s); keep clicking to add/remove, Esc to finish.",
                )
            )
            return
        if sketch_count > 0:
            self._material_face_hint.setText(
                self._ui(
                    f"已选 {sketch_count} 个草图区域；可继续点选增减，Esc 结束。",
                    f"Selected {sketch_count} sketch region(s); keep clicking to add/remove, Esc to finish.",
                )
            )
            return
        self._material_face_hint.setText(
            self._ui("等待在画布中点选区域（支持连续多选）...", "Waiting for canvas picks (continuous multi-select enabled)...")
        )

    def _update_component_region_selection_hint(self) -> None:
        region_count = len(self._selected_scene_region_ids_for_component_assignment())
        element_count = len(self._selected_face_region_element_ids)
        sketch_count = len(self._selected_face_region_sketch_keys)
        if element_count > 0:
            self._component_face_hint.setText(
                self._ui(
                    f"已选 {element_count} 个单元，映射 {region_count} 个区域；可继续点选增减，Esc 结束。",
                    f"Selected {element_count} elements mapped to {region_count} region(s); keep clicking to add/remove, Esc to finish.",
                )
            )
            return
        if sketch_count > 0:
            self._component_face_hint.setText(
                self._ui(
                    f"已选 {sketch_count} 个草图区域；可继续点选增减，Esc 结束。",
                    f"Selected {sketch_count} sketch region(s); keep clicking to add/remove, Esc to finish.",
                )
            )
            return
        self._component_face_hint.setText(
            self._ui("等待在画布中点选区域（支持连续多选）...", "Waiting for canvas picks (continuous multi-select enabled)...")
        )

    def _start_pick_component_face_region(self) -> None:
        if self._pending_pick_context in {"component_face", "component_face_sketch"}:
            self._pending_pick_context = None
            self._component_face_hint.setText(self._tr("rock.assembly.component.pick.exit", "已退出连续选区"))
            self._sync_pickable_sketch_faces_to_canvas()
            return
        self._scene_selection.active_region_ids.clear()
        has_mesh_elements = self._model is not None and bool(self._model.mesh.elements)
        if has_mesh_elements:
            self._pending_pick_context = "component_face"
            self._selected_face_region_element_ids.clear()
            self._selected_face_region_sketch_keys.clear()
            self._selected_element_ids.clear()
            self._mesh_canvas.set_highlighted_elements(set())
            self._mesh_canvas.set_highlighted_sketch_faces([])
            self._component_face_hint.setText(
                self._ui("等待在画布中点选区域（支持连续多选）...", "Waiting for canvas picks (continuous multi-select enabled)...")
            )
            self._log(
                self._ui(
                    "组件选区：请选择一个单元，系统将自动选择连通区域；可连续点选多个区域，Esc 结束。",
                    "Component pick: choose an element and a connected region will be selected; keep clicking to add/remove and press Esc to finish.",
                )
            )
        else:
            sketch_regions = self._collect_sketch_face_regions()
            if not sketch_regions:
                self._show_error(
                    self._ui("选择失败", "Pick failed"),
                    self._ui("当前没有可选单元或封闭草图区域。", "No mesh elements or closed sketch faces available."),
                )
                return
            self._pending_pick_context = "component_face_sketch"
            self._selected_face_region_element_ids.clear()
            self._selected_face_region_sketch_keys.clear()
            self._selected_element_ids.clear()
            self._mesh_canvas.set_highlighted_elements(set())
            self._mesh_canvas.set_highlighted_sketch_faces([])
            self._component_face_hint.setText(
                self._ui("等待在画布中点选草图面区域（支持连续多选）...", "Waiting for sketch-face picks (continuous multi-select enabled)...")
            )
            self._sync_pickable_sketch_faces_to_canvas()
            self._log(
                self._ui(
                    f"组件选区：已拆分 {len(sketch_regions)} 个原子子区域，请继续点选；Esc 结束。",
                    f"Component pick: {len(sketch_regions)} atomic sketch regions are available; keep clicking to add/remove and press Esc to finish.",
                )
            )
        if has_mesh_elements:
            self._sync_pickable_sketch_faces_to_canvas()
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("select")

    def _next_geometry_set_id(self) -> str:
        max_idx = 0
        for set_id in self._scene_project.geometry_sets:
            token = str(set_id).rsplit(":", 1)[-1]
            if token.isdigit():
                max_idx = max(max_idx, int(token))
        return f"set:{max_idx + 1}"

    def _next_manual_geometry_point_id(self) -> str:
        max_idx = 0
        for point_id in self._scene_project.geometry_points:
            token = str(point_id).rsplit(":", 1)[-1]
            if token.isdigit():
                max_idx = max(max_idx, int(token))
        return f"gpoint:manual:{max_idx + 1}"

    def _refresh_load_bc_target_set_combo(self, preferred_set_id: str | None = None) -> None:
        if not hasattr(self, "_load_bc_target_set_combo"):
            return
        combo = self._load_bc_target_set_combo
        current_set_id = preferred_set_id
        if current_set_id is None:
            current_data = combo.currentData()
            current_set_id = str(current_data) if isinstance(current_data, str) else None
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(self._ui("无（按当前参数）", "None (use current controls)"), "")
        for set_id in sorted(self._scene_project.geometry_sets):
            item = self._scene_project.geometry_sets[set_id]
            if item.entity_type not in {"point", "edge", "region", "component"}:
                continue
            item_type = item.entity_type
            if item_type == "point":
                kind_text = self._ui("点集", "Point")
            elif item_type == "edge":
                kind_text = self._ui("边集", "Edge")
            elif item_type == "region":
                kind_text = self._ui("区域集", "Region")
            else:
                kind_text = self._ui("组件集", "Component")
            mode_text = self._ui("几何" if item.binding_mode == "geometry" else "网格", "Geometry" if item.binding_mode == "geometry" else "Mesh")
            combo.addItem(f"{item.name} [{kind_text}/{mode_text}:{len(item.entity_ids)}]", set_id)
        idx = combo.findData(current_set_id) if current_set_id else -1
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        self._on_load_bc_target_set_changed(combo.currentIndex())
        self._refresh_load_bc_managers()
        self._refresh_results_scope_targets()

    def _selected_load_definition_id(self) -> str | None:
        if not hasattr(self, "_load_manager_table"):
            return None
        row = self._load_manager_table.currentRow()
        if row < 0:
            return None
        item = self._load_manager_table.item(row, 0)
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        return str(data) if isinstance(data, str) and data else None

    def _selected_boundary_definition_id(self) -> str | None:
        if not hasattr(self, "_bc_manager_table"):
            return None
        row = self._bc_manager_table.currentRow()
        if row < 0:
            return None
        item = self._bc_manager_table.item(row, 0)
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        return str(data) if isinstance(data, str) and data else None

    def _definition_status_text(self, *, definition_id: str, active: bool, kind: str) -> str:
        if not active:
            return self._ui("抑制", "Suppressed")
        if kind == "load" and definition_id in self._resolved_scene_loads_by_def_id:
            return self._ui(f"已解析 {len(self._resolved_scene_loads_by_def_id[definition_id])}", f"Resolved {len(self._resolved_scene_loads_by_def_id[definition_id])}")
        if kind == "bc" and definition_id in self._resolved_scene_bcs_by_def_id:
            return self._ui(f"已解析 {len(self._resolved_scene_bcs_by_def_id[definition_id])}", f"Resolved {len(self._resolved_scene_bcs_by_def_id[definition_id])}")
        return self._ui("待网格/待解析", "Pending")

    def _load_type_label(self, load_type: str) -> str:
        return {
            "concentrated": self._ui("集中力", "Point Force"),
            "distributed": self._ui("边线载", "Line Load"),
            "pressure": self._ui("边压力", "Edge Pressure"),
            "body": self._ui("区域体力", "Body Force"),
            "gravity": self._ui("自重", "Gravity"),
        }.get(str(load_type), str(load_type))

    def _target_set_label(self, set_id: str) -> str:
        target_set = self._scene_project.geometry_sets.get(str(set_id))
        if target_set is None:
            return str(set_id)
        mode_text = self._ui("几何" if target_set.binding_mode == "geometry" else "网格", "Geometry" if target_set.binding_mode == "geometry" else "Mesh")
        return f"{target_set.name} [{target_set.entity_type}/{mode_text}:{len(target_set.entity_ids)}]"

    def _load_target_entity_types(self, load_type: str) -> set[str]:
        if load_type == "concentrated":
            return {"point"}
        if load_type in {"distributed", "pressure"}:
            return {"edge"}
        if load_type in {"body", "gravity"}:
            return {"region", "component"}
        return {"point", "edge", "region", "component"}

    def _populate_target_set_combo(
        self,
        combo: QComboBox,
        *,
        entity_types: set[str],
        current_set_id: str | None = None,
    ) -> None:
        combo.clear()
        combo.addItem(self._ui("无（使用当前目标）", "None (use current target)"), "")
        for set_id in sorted(self._scene_project.geometry_sets):
            item = self._scene_project.geometry_sets[set_id]
            if item.entity_type not in entity_types:
                continue
            combo.addItem(self._target_set_label(set_id), set_id)
        if current_set_id:
            idx = combo.findData(current_set_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _next_default_load_name(self, load_type: str) -> str:
        prefix = {
            "concentrated": self._ui("集中力", "PointForce"),
            "distributed": self._ui("边线载", "LineLoad"),
            "pressure": self._ui("边压力", "Pressure"),
            "body": self._ui("区域体力", "BodyForce"),
            "gravity": self._ui("自重", "Gravity"),
        }.get(str(load_type), self._ui("载荷", "Load"))
        return f"{prefix}-{len(self._scene_project.load_definitions) + 1}"

    def _next_default_boundary_name(self) -> str:
        return self._ui(f"边界条件-{len(self._scene_project.boundary_definitions) + 1}", f"BC-{len(self._scene_project.boundary_definitions) + 1}")

    def _consume_pending_load_name(self, load_type: str) -> str:
        name = (self._pending_load_definition_name or "").strip()
        self._pending_load_definition_name = None
        return name or self._next_default_load_name(load_type)

    def _consume_pending_boundary_name(self) -> str:
        name = (self._pending_boundary_definition_name or "").strip()
        self._pending_boundary_definition_name = None
        return name or self._next_default_boundary_name()

    def _consume_pending_load_step(self) -> str:
        step = (self._pending_load_definition_step or "").strip()
        self._pending_load_definition_step = None
        return step or "Step-1"

    def _consume_pending_boundary_step(self) -> str:
        step = (self._pending_boundary_definition_step or "").strip()
        self._pending_boundary_definition_step = None
        return step or "Step-1"

    def _consume_pending_boundary_type(self) -> str:
        boundary_type = (self._pending_boundary_definition_type or "").strip()
        self._pending_boundary_definition_type = None
        return boundary_type or "displacement"

    def _boundary_direction_label(self, direction: str) -> str:
        return {"x": "U1", "y": "U2", "xy": "U1/U2"}.get(str(direction), str(direction).upper())

    def _boundary_type_label(self, boundary_type: str, direction: str) -> str:
        normalized = str(boundary_type or "displacement")
        if normalized == "fixed":
            return self._ui("完全固定", "Encastre")
        if normalized == "roller_x":
            return self._ui("滚动 X", "Roller X")
        if normalized == "roller_y":
            return self._ui("滚动 Y", "Roller Y")
        if normalized == "symmetry_x":
            return self._ui("X 对称", "X symmetry")
        if normalized == "symmetry_y":
            return self._ui("Y 对称", "Y symmetry")
        return self._ui(f"位移约束 {self._boundary_direction_label(direction)}", f"Displacement {self._boundary_direction_label(direction)}")

    def _refresh_load_bc_managers(self) -> None:
        if not hasattr(self, "_load_manager_table") or not hasattr(self, "_bc_manager_table"):
            return
        selected_load_id = self._selected_load_definition_id()
        selected_bc_id = self._selected_boundary_definition_id()
        load_items = sorted(self._scene_project.load_definitions.values(), key=lambda item: item.id)
        self._load_manager_table.blockSignals(True)
        self._load_manager_table.setRowCount(len(load_items))
        selected_load_row = -1
        for row, definition in enumerate(load_items):
            target_set = self._scene_project.geometry_sets.get(definition.target_set_id)
            target_name = target_set.name if target_set is not None else definition.target_set_id
            load_type = self._load_type_label(str(definition.load_type))
            values = [
                definition.name,
                load_type,
                f"{target_name} [{definition.target_entity_type}]",
                f"{float(definition.magnitude):.6g}",
                self._definition_status_text(definition_id=definition.id, active=bool(definition.active), kind="load"),
                definition.step,
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, definition.id)
                self._load_manager_table.setItem(row, col, item)
            if selected_load_id == definition.id:
                selected_load_row = row
        self._load_manager_table.blockSignals(False)
        if selected_load_row >= 0:
            self._load_manager_table.selectRow(selected_load_row)

        bc_items = sorted(self._scene_project.boundary_definitions.values(), key=lambda item: item.id)
        self._bc_manager_table.blockSignals(True)
        self._bc_manager_table.setRowCount(len(bc_items))
        selected_bc_row = -1
        for row, definition in enumerate(bc_items):
            target_set = self._scene_project.geometry_sets.get(definition.target_set_id)
            target_name = target_set.name if target_set is not None else definition.target_set_id
            dof_text = {"x": "UX", "y": "UY", "xy": "UX/UY"}.get(str(definition.direction), str(definition.direction))
            values = [
                definition.name,
                self._ui("位移约束", "Displacement"),
                f"{target_name} [{definition.target_entity_type}]",
                f"{dof_text}={float(definition.value):.6g}",
                self._definition_status_text(definition_id=definition.id, active=bool(definition.active), kind="bc"),
                definition.step,
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, definition.id)
                self._bc_manager_table.setItem(row, col, item)
            if selected_bc_id == definition.id:
                selected_bc_row = row
        self._bc_manager_table.blockSignals(False)
        if selected_bc_row >= 0:
            self._bc_manager_table.selectRow(selected_bc_row)

    def _refresh_after_load_bc_definition_change(self, reason: str) -> None:
        unresolved = self._resolve_scene_load_bc_definitions_to_model() if self._model is not None and self._model.mesh.nodes else []
        self._sync_load_bc_geometry_overlay()
        self._refresh_load_bc_managers()
        self._sync_editor_tables_from_model()
        self._update_model_tree()
        self._update_status_labels()
        self._mark_results_stale(reason)
        if unresolved:
            self._load_bc_set_hint.setText(
                self._ui(
                    "存在未解析的载荷/约束；生成保点网格或重新选择目标后会自动展开。",
                    "Some load/BC definitions are unresolved; generate a geometry-preserving mesh or re-pick targets.",
                )
            )

    def _open_create_load_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._ui("创建载荷", "Create Load"))
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name_edit = QLineEdit(self._next_default_load_name("concentrated"))
        form.addRow(self._ui("名称", "Name"), name_edit)
        step_edit = QLineEdit("Step-1")
        form.addRow(self._ui("分析步", "Step"), step_edit)
        load_type_combo = QComboBox()
        for label, data in (
            (self._ui("集中力", "Point Force"), "concentrated"),
            (self._ui("边线载", "Edge Line Load"), "distributed"),
            (self._ui("边压力", "Edge Pressure"), "pressure"),
            (self._ui("区域体力", "Region Body Force"), "body"),
            (self._ui("自重", "Gravity"), "gravity"),
        ):
            load_type_combo.addItem(label, data)
        form.addRow(self._ui("类型", "Type"), load_type_combo)
        layout.addLayout(form)
        hint = QLabel(
            self._ui(
                "继续后进入对应画布选择模式；选完目标后在右侧参数区确认并点击添加。",
                "Continue enters the matching canvas-pick mode; after picking, confirm parameters on the side panel and add.",
            )
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        continue_btn = QPushButton(self._ui("继续...", "Continue..."))
        cancel_btn = QPushButton(self._ui("取消", "Cancel"))
        buttons.addStretch(1)
        buttons.addWidget(continue_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        def accept_create() -> None:
            load_type = str(load_type_combo.currentData())
            self._pending_load_definition_name = name_edit.text().strip() or self._next_default_load_name(load_type)
            self._pending_load_definition_step = step_edit.text().strip() or "Step-1"
            idx = self._load_bc_target_mode_combo.findData("geometry")
            if idx >= 0:
                self._load_bc_target_mode_combo.setCurrentIndex(idx)
            if load_type == "concentrated":
                self._start_pick_geometry_point_target()
            elif load_type == "distributed":
                type_idx = self._load_dist_type_combo.findData("distributed")
                if type_idx >= 0:
                    self._load_dist_type_combo.setCurrentIndex(type_idx)
                self._start_pick_distributed_edge_target()
            elif load_type == "pressure":
                self._prepare_edge_pressure_creation()
            elif load_type == "gravity":
                self._prepare_gravity_creation()
            else:
                self._prepare_region_body_force_creation()
            self._load_bc_set_hint.setText(
                self._ui(
                    f"正在创建 {self._pending_load_definition_name}：请选择目标，然后输入参数并点击添加。",
                    f"Creating {self._pending_load_definition_name}: pick a target, enter parameters, then add.",
                )
            )
            dialog.accept()

        continue_btn.clicked.connect(accept_create)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _open_create_boundary_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._ui("创建边界条件", "Create Boundary Condition"))
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name_edit = QLineEdit(self._next_default_boundary_name())
        form.addRow(self._ui("名称", "Name"), name_edit)
        step_edit = QLineEdit("Step-1")
        form.addRow(self._ui("分析步", "Step"), step_edit)
        target_combo = QComboBox()
        target_combo.addItem(self._ui("点", "Point"), "point")
        target_combo.addItem(self._ui("边", "Edge"), "edge")
        form.addRow(self._ui("区域类型", "Target"), target_combo)
        bc_type_combo = QComboBox()
        for label, data in (
            (self._ui("完全固定 UX/UY", "Fixed UX/UY"), "xy"),
            (self._ui("约束 UX", "Fix UX"), "x"),
            (self._ui("约束 UY", "Fix UY"), "y"),
            (self._ui("滚动 X（约束 UX）", "Roller X (fix UX)"), "x"),
            (self._ui("滚动 Y（约束 UY）", "Roller Y (fix UY)"), "y"),
        ):
            bc_type_combo.addItem(label, data)
        form.addRow(self._ui("类型", "Type"), bc_type_combo)
        layout.addLayout(form)
        hint = QLabel(
            self._ui(
                "继续后进入点/边选择模式；选完目标后点击“应用约束”。二维实体只支持 UX/UY 平动自由度。",
                "Continue enters point/edge pick mode; after picking, click Apply BC. 2D continuum supports UX/UY translational DOFs.",
            )
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        continue_btn = QPushButton(self._ui("继续...", "Continue..."))
        cancel_btn = QPushButton(self._ui("取消", "Cancel"))
        buttons.addStretch(1)
        buttons.addWidget(continue_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        def accept_create() -> None:
            self._pending_boundary_definition_name = name_edit.text().strip() or self._next_default_boundary_name()
            self._pending_boundary_definition_step = step_edit.text().strip() or "Step-1"
            self._set_bc_direction_and_value(str(bc_type_combo.currentData()), 0.0)
            idx = self._load_bc_target_mode_combo.findData("geometry")
            if idx >= 0:
                self._load_bc_target_mode_combo.setCurrentIndex(idx)
            target_idx = self._bc_target_combo.findData(str(target_combo.currentData()))
            if target_idx >= 0:
                self._bc_target_combo.setCurrentIndex(target_idx)
            if str(target_combo.currentData()) == "edge":
                self._start_pick_bc_edge()
            else:
                self._start_pick_geometry_point_target()
            self._load_bc_set_hint.setText(
                self._ui(
                    f"正在创建 {self._pending_boundary_definition_name}：请选择目标，然后点击应用约束。",
                    f"Creating {self._pending_boundary_definition_name}: pick a target, then apply the BC.",
                )
            )
            dialog.accept()

        continue_btn.clicked.connect(accept_create)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _rename_load_definition(self, definition_id: str) -> None:
        definition = self._scene_project.load_definitions.get(str(definition_id))
        if definition is None:
            return
        name, ok = QInputDialog.getText(
            self,
            self._ui("重命名载荷", "Rename Load"),
            self._ui("新名称", "New name"),
            text=definition.name,
        )
        if not ok or not name.strip():
            return
        definition.name = name.strip()
        self._refresh_after_load_bc_definition_change("Load definition renamed")

    def _rename_boundary_definition(self, definition_id: str) -> None:
        definition = self._scene_project.boundary_definitions.get(str(definition_id))
        if definition is None:
            return
        name, ok = QInputDialog.getText(
            self,
            self._ui("重命名边界条件", "Rename Boundary Condition"),
            self._ui("新名称", "New name"),
            text=definition.name,
        )
        if not ok or not name.strip():
            return
        definition.name = name.strip()
        self._refresh_after_load_bc_definition_change("Boundary definition renamed")

    def _copy_load_definition(self, definition_id: str) -> str | None:
        definition = self._scene_project.load_definitions.get(str(definition_id))
        if definition is None:
            return None
        new_id = self._next_load_definition_id()
        copied = copy.deepcopy(definition)
        copied.id = new_id
        copied.name = f"{definition.name}-Copy"
        self._scene_project.load_definitions[new_id] = copied
        self._refresh_after_load_bc_definition_change("Load definition copied")
        return new_id

    def _copy_boundary_definition(self, definition_id: str) -> str | None:
        definition = self._scene_project.boundary_definitions.get(str(definition_id))
        if definition is None:
            return None
        new_id = self._next_boundary_definition_id()
        copied = copy.deepcopy(definition)
        copied.id = new_id
        copied.name = f"{definition.name}-Copy"
        self._scene_project.boundary_definitions[new_id] = copied
        self._refresh_after_load_bc_definition_change("Boundary definition copied")
        return new_id

    def _open_edit_load_definition_dialog(self, definition_id: str) -> None:
        definition = self._scene_project.load_definitions.get(str(definition_id))
        if definition is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self._ui("编辑载荷", "Edit Load"))
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name_edit = QLineEdit(definition.name)
        form.addRow(self._ui("名称", "Name"), name_edit)
        type_label = QLabel(self._load_type_label(str(definition.load_type)))
        form.addRow(self._ui("类型", "Type"), type_label)
        step_edit = QLineEdit(definition.step)
        form.addRow(self._ui("分析步", "Step"), step_edit)
        target_combo = QComboBox()
        self._populate_target_set_combo(
            target_combo,
            entity_types=self._load_target_entity_types(str(definition.load_type)),
            current_set_id=definition.target_set_id,
        )
        form.addRow(self._ui("目标集合", "Target Set"), target_combo)
        vector_x = QDoubleSpinBox()
        vector_x.setDecimals(6)
        vector_x.setRange(-1e12, 1e12)
        vector_x.setValue(float(definition.vector_x))
        form.addRow(self._ui("方向 X", "Dir X"), vector_x)
        vector_y = QDoubleSpinBox()
        vector_y.setDecimals(6)
        vector_y.setRange(-1e12, 1e12)
        vector_y.setValue(float(definition.vector_y))
        form.addRow(self._ui("方向 Y", "Dir Y"), vector_y)
        magnitude = QDoubleSpinBox()
        magnitude.setDecimals(6)
        magnitude.setRange(-1e18, 1e18)
        magnitude.setValue(float(definition.magnitude))
        form.addRow(self._ui("大小", "Magnitude"), magnitude)
        direction_mode = QComboBox()
        for label, data in (
            (self._ui("自定义向量", "Custom Vector"), "vector"),
            (self._ui("边法向", "Edge Normal"), "normal"),
            (self._ui("反向法向", "Reverse Normal"), "reverse_normal"),
        ):
            direction_mode.addItem(label, data)
        mode_idx = direction_mode.findData(getattr(definition, "direction_mode", "vector"))
        if mode_idx >= 0:
            direction_mode.setCurrentIndex(mode_idx)
        direction_mode.setEnabled(definition.load_type == "pressure")
        form.addRow(self._ui("方向模式", "Direction Mode"), direction_mode)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        ok_btn = QPushButton(self._ui("确定", "OK"))
        cancel_btn = QPushButton(self._ui("取消", "Cancel"))
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        def apply_edit() -> None:
            definition.name = name_edit.text().strip() or definition.name
            definition.step = step_edit.text().strip() or definition.step
            selected_set_id = str(target_combo.currentData() or "")
            target_set = self._scene_project.geometry_sets.get(selected_set_id)
            if target_set is not None:
                definition.target_set_id = target_set.id
                definition.target_entity_type = target_set.entity_type
            definition.vector_x = float(vector_x.value())
            definition.vector_y = float(vector_y.value())
            definition.magnitude = float(magnitude.value())
            definition.direction_mode = str(direction_mode.currentData())
            self._refresh_after_load_bc_definition_change("Load definition edited")
            dialog.accept()

        ok_btn.clicked.connect(apply_edit)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _open_edit_boundary_definition_dialog(self, definition_id: str) -> None:
        definition = self._scene_project.boundary_definitions.get(str(definition_id))
        if definition is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self._ui("编辑边界条件", "Edit Boundary Condition"))
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name_edit = QLineEdit(definition.name)
        form.addRow(self._ui("名称", "Name"), name_edit)
        step_edit = QLineEdit(definition.step)
        form.addRow(self._ui("分析步", "Step"), step_edit)
        target_combo = QComboBox()
        self._populate_target_set_combo(target_combo, entity_types={"point", "edge"}, current_set_id=definition.target_set_id)
        form.addRow(self._ui("目标集合", "Target Set"), target_combo)
        direction_combo = QComboBox()
        direction_combo.addItem("UX", "x")
        direction_combo.addItem("UY", "y")
        direction_combo.addItem("UX/UY", "xy")
        direction_idx = direction_combo.findData(definition.direction)
        if direction_idx >= 0:
            direction_combo.setCurrentIndex(direction_idx)
        form.addRow(self._ui("自由度", "DOF"), direction_combo)
        value_spin = QDoubleSpinBox()
        value_spin.setDecimals(6)
        value_spin.setRange(-1e12, 1e12)
        value_spin.setValue(float(definition.value))
        form.addRow(self._ui("值", "Value"), value_spin)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        ok_btn = QPushButton(self._ui("确定", "OK"))
        cancel_btn = QPushButton(self._ui("取消", "Cancel"))
        buttons.addStretch(1)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        def apply_edit() -> None:
            definition.name = name_edit.text().strip() or definition.name
            definition.step = step_edit.text().strip() or definition.step
            selected_set_id = str(target_combo.currentData() or "")
            target_set = self._scene_project.geometry_sets.get(selected_set_id)
            if target_set is not None and target_set.entity_type in {"point", "edge"}:
                definition.target_set_id = target_set.id
                definition.target_entity_type = target_set.entity_type
            definition.direction = str(direction_combo.currentData())
            definition.value = float(value_spin.value())
            self._refresh_after_load_bc_definition_change("Boundary definition edited")
            dialog.accept()

        ok_btn.clicked.connect(apply_edit)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _open_load_manager_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._ui("载荷管理器", "Load Manager"))
        dialog.resize(780, 460)
        layout = QVBoxLayout(dialog)
        content = QHBoxLayout()
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels([
            self._ui("名称", "Name"),
            self._ui("Step-1", "Step-1"),
            self._ui("类型", "Type"),
            self._ui("区域", "Region"),
            self._ui("状态", "Status"),
        ])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        content.addWidget(table)
        side = QVBoxLayout()
        edit_btn = QPushButton(self._ui("编辑...", "Edit..."))
        copy_btn = QPushButton(self._ui("复制...", "Copy..."))
        rename_btn = QPushButton(self._ui("重命名...", "Rename..."))
        delete_btn = QPushButton(self._ui("删除...", "Delete..."))
        toggle_btn = QPushButton(self._ui("激活/抑制", "Activate/Suppress"))
        locate_btn = QPushButton(self._ui("定位", "Locate"))
        for button in (edit_btn, copy_btn, rename_btn, delete_btn, toggle_btn, locate_btn):
            side.addWidget(button)
        side.addStretch(1)
        content.addLayout(side)
        layout.addLayout(content)
        detail = QLabel()
        detail.setWordWrap(True)
        layout.addWidget(detail)
        bottom = QHBoxLayout()
        create_btn = QPushButton(self._ui("创建...", "Create..."))
        close_btn = QPushButton(self._ui("关闭", "Close"))
        bottom.addWidget(create_btn)
        bottom.addStretch(1)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

        def selected_id() -> str | None:
            row = table.currentRow()
            if row < 0:
                return None
            item = table.item(row, 0)
            if item is None:
                return None
            data = item.data(Qt.UserRole)
            return str(data) if isinstance(data, str) else None

        def refresh() -> None:
            items = sorted(self._scene_project.load_definitions.values(), key=lambda item: item.id)
            table.setRowCount(len(items))
            for row, definition in enumerate(items):
                values = [
                    definition.name,
                    definition.step,
                    self._load_type_label(str(definition.load_type)),
                    self._target_set_label(definition.target_set_id),
                    self._definition_status_text(definition_id=definition.id, active=bool(definition.active), kind="load"),
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.UserRole, definition.id)
                    table.setItem(row, col, item)
            if items and table.currentRow() < 0:
                table.selectRow(0)

        def update_detail() -> None:
            definition = self._scene_project.load_definitions.get(selected_id() or "")
            if definition is None:
                detail.setText(self._ui("未选择载荷。", "No load selected."))
                return
            detail.setText(
                self._ui(
                    f"分析步: {definition.step}\n载荷 类型: {self._load_type_label(str(definition.load_type))}\n载荷 状态: {'已创建' if definition.active else '已抑制'}",
                    f"Step: {definition.step}\nLoad type: {self._load_type_label(str(definition.load_type))}\nStatus: {'Created' if definition.active else 'Suppressed'}",
                )
            )

        def refresh_all() -> None:
            refresh()
            self._refresh_load_bc_managers()
            update_detail()

        table.itemSelectionChanged.connect(update_detail)
        edit_btn.clicked.connect(lambda: (self._open_edit_load_definition_dialog(selected_id() or ""), refresh_all()))
        copy_btn.clicked.connect(lambda: (self._copy_load_definition(selected_id() or ""), refresh_all()))
        rename_btn.clicked.connect(lambda: (self._rename_load_definition(selected_id() or ""), refresh_all()))
        delete_btn.clicked.connect(lambda: (self._scene_project.load_definitions.pop(selected_id() or "", None), self._refresh_after_load_bc_definition_change("Load definition deleted"), refresh_all()))
        toggle_btn.clicked.connect(lambda: (setattr(self._scene_project.load_definitions[selected_id() or ""], "active", not self._scene_project.load_definitions[selected_id() or ""].active) if selected_id() in self._scene_project.load_definitions else None, self._refresh_after_load_bc_definition_change("Load definition activation changed"), refresh_all()))
        locate_btn.clicked.connect(lambda: self._locate_load_bc_target_set(self._scene_project.load_definitions[selected_id() or ""].target_set_id) if selected_id() in self._scene_project.load_definitions else None)
        create_btn.clicked.connect(lambda: (dialog.accept(), self._open_create_load_dialog()))
        close_btn.clicked.connect(dialog.accept)
        refresh_all()
        dialog.exec()

    def _open_boundary_manager_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._ui("边界条件管理器", "Boundary Condition Manager"))
        dialog.resize(780, 460)
        layout = QVBoxLayout(dialog)
        content = QHBoxLayout()
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels([
            self._ui("名称", "Name"),
            self._ui("Initial", "Initial"),
            self._ui("Step-1", "Step-1"),
            self._ui("类型", "Type"),
            self._ui("状态", "Status"),
        ])
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        content.addWidget(table)
        side = QVBoxLayout()
        edit_btn = QPushButton(self._ui("编辑...", "Edit..."))
        copy_btn = QPushButton(self._ui("复制...", "Copy..."))
        rename_btn = QPushButton(self._ui("重命名...", "Rename..."))
        delete_btn = QPushButton(self._ui("删除...", "Delete..."))
        toggle_btn = QPushButton(self._ui("激活/抑制", "Activate/Suppress"))
        locate_btn = QPushButton(self._ui("定位", "Locate"))
        for button in (edit_btn, copy_btn, rename_btn, delete_btn, toggle_btn, locate_btn):
            side.addWidget(button)
        side.addStretch(1)
        content.addLayout(side)
        layout.addLayout(content)
        detail = QLabel()
        detail.setWordWrap(True)
        layout.addWidget(detail)
        bottom = QHBoxLayout()
        create_btn = QPushButton(self._ui("创建...", "Create..."))
        close_btn = QPushButton(self._ui("关闭", "Close"))
        bottom.addWidget(create_btn)
        bottom.addStretch(1)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

        def selected_id() -> str | None:
            row = table.currentRow()
            if row < 0:
                return None
            item = table.item(row, 0)
            if item is None:
                return None
            data = item.data(Qt.UserRole)
            return str(data) if isinstance(data, str) else None

        def refresh() -> None:
            items = sorted(self._scene_project.boundary_definitions.values(), key=lambda item: item.id)
            table.setRowCount(len(items))
            for row, definition in enumerate(items):
                values = [
                    definition.name,
                    self._ui("已创建", "Created") if definition.step.lower() == "initial" else self._ui("传递", "Propagated"),
                    self._ui("已创建", "Created") if definition.step.lower() != "initial" else self._ui("传递", "Propagated"),
                    self._ui("位移/完全固定", "Displacement/Fixed"),
                    self._definition_status_text(definition_id=definition.id, active=bool(definition.active), kind="bc"),
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.UserRole, definition.id)
                    table.setItem(row, col, item)
            if items and table.currentRow() < 0:
                table.selectRow(0)

        def update_detail() -> None:
            definition = self._scene_project.boundary_definitions.get(selected_id() or "")
            if definition is None:
                detail.setText(self._ui("未选择边界条件。", "No boundary condition selected."))
                return
            dof_text = {"x": "UX", "y": "UY", "xy": "UX/UY"}.get(str(definition.direction), str(definition.direction))
            detail.setText(
                self._ui(
                    f"分析步: {definition.step}\n边界条件 类型: 位移/完全固定\n目标: {self._target_set_label(definition.target_set_id)}\n自由度: {dof_text}",
                    f"Step: {definition.step}\nBC type: Displacement/Fixed\nTarget: {self._target_set_label(definition.target_set_id)}\nDOF: {dof_text}",
                )
            )

        def refresh_all() -> None:
            refresh()
            self._refresh_load_bc_managers()
            update_detail()

        table.itemSelectionChanged.connect(update_detail)
        edit_btn.clicked.connect(lambda: (self._open_edit_boundary_definition_dialog(selected_id() or ""), refresh_all()))
        copy_btn.clicked.connect(lambda: (self._copy_boundary_definition(selected_id() or ""), refresh_all()))
        rename_btn.clicked.connect(lambda: (self._rename_boundary_definition(selected_id() or ""), refresh_all()))
        delete_btn.clicked.connect(lambda: (self._scene_project.boundary_definitions.pop(selected_id() or "", None), self._refresh_after_load_bc_definition_change("Boundary definition deleted"), refresh_all()))
        toggle_btn.clicked.connect(lambda: (setattr(self._scene_project.boundary_definitions[selected_id() or ""], "active", not self._scene_project.boundary_definitions[selected_id() or ""].active) if selected_id() in self._scene_project.boundary_definitions else None, self._refresh_after_load_bc_definition_change("Boundary definition activation changed"), refresh_all()))
        locate_btn.clicked.connect(lambda: self._locate_load_bc_target_set(self._scene_project.boundary_definitions[selected_id() or ""].target_set_id) if selected_id() in self._scene_project.boundary_definitions else None)
        create_btn.clicked.connect(lambda: (dialog.accept(), self._open_create_boundary_dialog()))
        close_btn.clicked.connect(dialog.accept)
        refresh_all()
        dialog.exec()

    def _locate_load_bc_target_set(self, set_id: str) -> None:
        idx = self._load_bc_target_set_combo.findData(set_id)
        if idx >= 0:
            self._load_bc_target_set_combo.setCurrentIndex(idx)
        target_set = self._scene_project.geometry_sets.get(set_id)
        if target_set is None:
            return
        self._selected_geometry_point_ids.clear()
        self._selected_geometry_edge_ids.clear()
        if target_set.binding_mode == "geometry":
            if target_set.entity_type == "point":
                self._selected_geometry_point_ids = {str(item) for item in target_set.entity_ids}
            elif target_set.entity_type == "edge":
                self._selected_geometry_edge_ids = {str(item) for item in target_set.entity_ids}
            elif target_set.entity_type in {"region", "component"}:
                element_ids = set(self._scene_mesh_state.set_to_element_ids.get(set_id, []))
                self._mesh_canvas.set_highlighted_elements(element_ids)
        else:
            node_ids = set(self._scene_mesh_state.set_to_node_ids.get(set_id, []))
            self._mesh_canvas.set_highlighted_nodes(node_ids)
        self._sync_load_bc_geometry_overlay()

    def _locate_selected_load_definition(self) -> None:
        definition_id = self._selected_load_definition_id()
        definition = self._scene_project.load_definitions.get(definition_id or "")
        if definition is None:
            return
        self._locate_load_bc_target_set(definition.target_set_id)

    def _locate_selected_boundary_definition(self) -> None:
        definition_id = self._selected_boundary_definition_id()
        definition = self._scene_project.boundary_definitions.get(definition_id or "")
        if definition is None:
            return
        self._locate_load_bc_target_set(definition.target_set_id)

    def _toggle_selected_load_definition_active(self) -> None:
        definition_id = self._selected_load_definition_id()
        definition = self._scene_project.load_definitions.get(definition_id or "")
        if definition is None:
            return
        definition.active = not bool(definition.active)
        self._refresh_after_load_bc_definition_change("Load definition activation changed")

    def _toggle_selected_boundary_definition_active(self) -> None:
        definition_id = self._selected_boundary_definition_id()
        definition = self._scene_project.boundary_definitions.get(definition_id or "")
        if definition is None:
            return
        definition.active = not bool(definition.active)
        self._refresh_after_load_bc_definition_change("Boundary definition activation changed")

    def _delete_selected_load_definition(self) -> None:
        definition_id = self._selected_load_definition_id()
        if not definition_id:
            return
        self._scene_project.load_definitions.pop(definition_id, None)
        self._refresh_after_load_bc_definition_change("Load definition deleted")

    def _delete_selected_boundary_definition(self) -> None:
        definition_id = self._selected_boundary_definition_id()
        if not definition_id:
            return
        self._scene_project.boundary_definitions.pop(definition_id, None)
        self._refresh_after_load_bc_definition_change("Boundary definition deleted")

    def _update_selected_load_definition_from_controls(self) -> None:
        definition_id = self._selected_load_definition_id()
        definition = self._scene_project.load_definitions.get(definition_id or "")
        if definition is None:
            return
        target_set = self._current_load_bc_target_set()
        if definition.load_type == "concentrated":
            if target_set is not None and target_set.entity_type == "point":
                definition.target_set_id = target_set.id
                definition.target_entity_type = "point"
            definition.vector_x = float(self._load_point_vec_x.value())
            definition.vector_y = float(self._load_point_vec_y.value())
            definition.magnitude = float(self._load_point_magnitude_spin.value())
            definition.direction_mode = "vector"
        elif definition.load_type in {"distributed", "pressure"}:
            if target_set is not None and target_set.entity_type == "edge":
                definition.target_set_id = target_set.id
                definition.target_entity_type = "edge"
            definition.load_type = "pressure" if str(self._load_dist_type_combo.currentData()) == "pressure" else "distributed"
            definition.vector_x = float(self._load_dist_vec_x.value())
            definition.vector_y = float(self._load_dist_vec_y.value())
            definition.magnitude = float(self._load_dist_total_mag_spin.value())
            definition.profile = "linear" if str(self._load_dist_profile_combo.currentData()) == "linear" else "uniform"
            definition.direction_mode = str(self._load_dist_direction_mode_combo.currentData())
        elif definition.load_type in {"body", "gravity"}:
            if target_set is not None and target_set.entity_type in {"region", "component"}:
                definition.target_set_id = target_set.id
                definition.target_entity_type = target_set.entity_type
            definition.vector_x = float(self._region_load_vec_x.value())
            definition.vector_y = float(self._region_load_vec_y.value())
            definition.magnitude = float(self._region_load_intensity_spin.value())
        self._refresh_after_load_bc_definition_change("Load definition edited")

    def _update_selected_boundary_definition_from_controls(self) -> None:
        definition_id = self._selected_boundary_definition_id()
        definition = self._scene_project.boundary_definitions.get(definition_id or "")
        if definition is None:
            return
        target_set = self._current_load_bc_target_set()
        if target_set is not None and target_set.entity_type in {"point", "edge"}:
            definition.target_set_id = target_set.id
            definition.target_entity_type = target_set.entity_type
        direction = str(self._bc_dir_combo.currentData())
        definition.direction = "x" if direction == "x" else "y" if direction == "y" else "xy"
        definition.value = float(self._bc_value_spin.value())
        self._refresh_after_load_bc_definition_change("Boundary definition edited")

    def _selected_load_bc_set_node_ids(self) -> list[int]:
        if self._model is None or not hasattr(self, "_load_bc_target_set_combo"):
            return []
        set_id = self._load_bc_target_set_combo.currentData()
        if not isinstance(set_id, str) or not set_id:
            return []
        node_ids = self._scene_mesh_state.set_to_node_ids.get(set_id, [])
        return sorted({int(node_id) for node_id in node_ids})

    def _upsert_geometry_set_from_node_ids(
        self,
        *,
        entity_type: str,
        node_ids: list[int] | set[int],
        preferred_name: str,
    ) -> str | None:
        normalized = sorted({int(node_id) for node_id in node_ids if int(node_id) > 0})
        if not normalized:
            return None
        entity_ids = [str(node_id) for node_id in normalized]
        existing_id: str | None = None
        for set_id, item in self._scene_project.geometry_sets.items():
            if item.entity_type != entity_type:
                continue
            if item.binding_mode != "mesh":
                continue
            if list(item.entity_ids) == entity_ids:
                existing_id = set_id
                break
        if existing_id is not None:
            target = self._scene_project.geometry_sets[existing_id]
            if preferred_name.strip():
                target.name = preferred_name.strip()
            self._scene_mesh_state.set_to_node_ids[existing_id] = normalized
            self._refresh_load_bc_target_set_combo(existing_id)
            return existing_id
        set_id = self._next_geometry_set_id()
        self._scene_project.geometry_sets[set_id] = GeometrySetDef(
            id=set_id,
            name=preferred_name.strip() or set_id,
            entity_type="point" if entity_type == "point" else "edge",
            entity_ids=entity_ids,
            binding_mode="mesh",
            auto_update=False,
            display_color="#f59e0b",
        )
        self._scene_mesh_state.set_to_node_ids[set_id] = normalized
        self._refresh_load_bc_target_set_combo(set_id)
        return set_id

    def _refresh_scene_mesh_state_for_set(self, set_id: str) -> None:
        if not set_id:
            return
        geometry_set = self._scene_project.geometry_sets.get(set_id)
        if geometry_set is None:
            self._scene_mesh_state.set_to_node_ids.pop(set_id, None)
            self._scene_mesh_state.set_to_element_ids.pop(set_id, None)
            return

        node_id_set = {int(node.id) for node in self._model.mesh.nodes} if self._model is not None else set()
        entity_type = str(geometry_set.entity_type)
        binding_mode = str(geometry_set.binding_mode)

        if entity_type == "point":
            node_ids: list[int] = []
            if binding_mode == "geometry":
                for point_id in geometry_set.entity_ids:
                    node_ids.extend(self._scene_mesh_state.point_to_node_ids.get(str(point_id), []))
            else:
                for token in geometry_set.entity_ids:
                    try:
                        node_id = int(token)
                    except (TypeError, ValueError):
                        continue
                    if node_id in node_id_set:
                        node_ids.append(node_id)
            self._scene_mesh_state.set_to_node_ids[set_id] = sorted(set(int(item) for item in node_ids if int(item) > 0))
            self._scene_mesh_state.set_to_element_ids.pop(set_id, None)
            return

        if entity_type == "edge":
            node_ids = []
            if binding_mode == "geometry":
                for edge_id in geometry_set.entity_ids:
                    node_ids.extend(self._scene_mesh_state.edge_to_node_ids.get(str(edge_id), []))
            else:
                for token in geometry_set.entity_ids:
                    try:
                        node_id = int(token)
                    except (TypeError, ValueError):
                        continue
                    if node_id in node_id_set:
                        node_ids.append(node_id)
            self._scene_mesh_state.set_to_node_ids[set_id] = sorted(set(int(item) for item in node_ids if int(item) > 0))
            self._scene_mesh_state.set_to_element_ids.pop(set_id, None)
            return

        if entity_type == "region":
            element_ids: list[int] = []
            for region_id in geometry_set.entity_ids:
                element_ids.extend(self._scene_mesh_state.region_to_element_ids.get(str(region_id), []))
            self._scene_mesh_state.set_to_element_ids[set_id] = sorted(set(int(item) for item in element_ids if int(item) > 0))
            self._scene_mesh_state.set_to_node_ids.pop(set_id, None)
            return

        if entity_type == "component":
            element_ids = []
            for component_id in geometry_set.entity_ids:
                element_ids.extend(self._scene_mesh_state.component_to_element_ids.get(str(component_id), []))
            self._scene_mesh_state.set_to_element_ids[set_id] = sorted(set(int(item) for item in element_ids if int(item) > 0))
            self._scene_mesh_state.set_to_node_ids.pop(set_id, None)
            return

        self._scene_mesh_state.set_to_node_ids.pop(set_id, None)
        self._scene_mesh_state.set_to_element_ids.pop(set_id, None)

    def _upsert_geometry_set_from_entities(
        self,
        *,
        entity_type: str,
        entity_ids: list[str] | set[str],
        preferred_name: str,
        binding_mode: str = "geometry",
        preferred_set_id: str | None = None,
    ) -> str | None:
        normalized = sorted({str(item) for item in entity_ids if str(item).strip()})
        if not normalized:
            return None
        if preferred_set_id:
            preferred = self._scene_project.geometry_sets.get(str(preferred_set_id))
            if (
                preferred is not None
                and preferred.entity_type == entity_type
                and str(preferred.binding_mode) == str(binding_mode)
            ):
                preferred.entity_ids = normalized
                if preferred_name.strip():
                    preferred.name = preferred_name.strip()
                self._refresh_scene_mesh_state_for_set(str(preferred_set_id))
                self._refresh_load_bc_target_set_combo(str(preferred_set_id))
                return str(preferred_set_id)
        existing_id: str | None = None
        for set_id, item in self._scene_project.geometry_sets.items():
            if item.entity_type != entity_type:
                continue
            if str(item.binding_mode) != str(binding_mode):
                continue
            if list(item.entity_ids) == normalized:
                existing_id = set_id
                break
        if existing_id is not None:
            target = self._scene_project.geometry_sets[existing_id]
            if preferred_name.strip():
                target.name = preferred_name.strip()
            self._refresh_scene_mesh_state_for_set(existing_id)
            self._refresh_load_bc_target_set_combo(existing_id)
            return existing_id
        set_id = self._next_geometry_set_id()
        self._scene_project.geometry_sets[set_id] = GeometrySetDef(
            id=set_id,
            name=preferred_name.strip() or set_id,
            entity_type=entity_type,
            entity_ids=normalized,
            binding_mode="mesh" if str(binding_mode) == "mesh" else "geometry",
            auto_update=False,
            display_color="#f59e0b",
        )
        self._refresh_scene_mesh_state_for_set(set_id)
        self._refresh_load_bc_target_set_combo(set_id)
        return set_id

    def _upsert_geometry_set_from_geometry_ids(
        self,
        *,
        entity_type: str,
        geometry_ids: list[str] | set[str],
        preferred_name: str,
        preferred_set_id: str | None = None,
    ) -> str | None:
        return self._upsert_geometry_set_from_entities(
            entity_type=entity_type,
            entity_ids=geometry_ids,
            preferred_name=preferred_name,
            binding_mode="geometry",
            preferred_set_id=preferred_set_id,
        )

    def _capture_geometry_set_from_current_target(self) -> None:
        target_type = str(self._bc_target_combo.currentData() or "point")
        if self._current_load_bc_target_mode() == "geometry":
            current_set = self._current_load_bc_target_set()
            preferred_point_set_id = (
                current_set.id
                if current_set is not None and current_set.binding_mode == "geometry" and current_set.entity_type == "point"
                else self._last_geometry_pick_set_id
            )
            preferred_edge_set_id = (
                current_set.id
                if current_set is not None and current_set.binding_mode == "geometry" and current_set.entity_type == "edge"
                else self._last_geometry_pick_set_id
            )
            if target_type == "point":
                set_id = self._upsert_geometry_set_from_geometry_ids(
                    entity_type="point",
                    geometry_ids=self._selected_geometry_point_ids,
                    preferred_name=self._ui("几何点集合", "Geometry Point Set"),
                    preferred_set_id=None if self._is_geometry_set_used_by_load_bc_definition(preferred_point_set_id) else preferred_point_set_id,
                )
            elif target_type == "edge":
                set_id = self._upsert_geometry_set_from_geometry_ids(
                    entity_type="edge",
                    geometry_ids=self._selected_geometry_edge_ids,
                    preferred_name=self._ui("几何边集合", "Geometry Edge Set"),
                    preferred_set_id=None if self._is_geometry_set_used_by_load_bc_definition(preferred_edge_set_id) else preferred_edge_set_id,
                )
            else:
                region_ids = [
                    self._scene_region_id_from_key(key)
                    for key in sorted(self._selected_face_region_sketch_keys)
                    if self._scene_region_id_from_key(key) in self._scene_project.regions
                ]
                region_ids.extend(sorted(self._scene_selection.active_region_ids))
                set_id = self._upsert_geometry_set_from_entities(
                    entity_type="region",
                    entity_ids=sorted(set(region_ids)),
                    preferred_name=self._ui("几何区域集合", "Geometry Region Set"),
                    binding_mode="geometry",
                )
            if set_id is None:
                self._show_error(
                    self._ui("集合创建失败", "Set capture failed"),
                    self._ui("当前没有可用几何目标，请先进行几何选点/选边。", "No geometry target selected. Pick geometry point/edge first."),
                )
                return
            self._last_geometry_pick_set_id = set_id
            self._log(
                self._ui(
                    f"已创建/更新几何集合 {set_id}。",
                    f"Captured/updated geometry set {set_id}.",
                )
            )
            return

        if self._model is None:
            return
        node_ids: list[int] = []
        entity_type = "point"
        if target_type == "point":
            node_id = int(self._bc_point_node_spin.value())
            if self._get_node_by_id(node_id) is not None:
                node_ids = [node_id]
            entity_type = "point"
        elif target_type == "edge":
            axis = str(self._bc_edge_axis_combo.currentData())
            coord = float(self._bc_edge_coord_spin.value())
            tol = float(self._bc_edge_tol_spin.value())
            for node in self._model.mesh.nodes:
                value_axis = node.x if axis == "x" else node.y
                if abs(value_axis - coord) <= tol:
                    node_ids.append(int(node.id))
            entity_type = "edge"
        else:
            node_ids = [int(node.id) for node in self._model.mesh.nodes]
            entity_type = "edge"
        set_id = self._upsert_geometry_set_from_node_ids(
            entity_type=entity_type,
            node_ids=node_ids,
            preferred_name=self._ui("目标集合", "Target Set"),
        )
        if set_id is None:
            self._show_error(
                self._ui("集合创建失败", "Set capture failed"),
                self._ui("当前参数没有匹配到可用节点。", "Current controls did not match any nodes."),
            )
            return
        self._log(
            self._ui(
                f"已创建/更新目标集合 {set_id}，节点数={len(node_ids)}。",
                f"Captured target set {set_id} with {len(node_ids)} nodes.",
            )
        )

    def _capture_region_set_from_selected_regions(self) -> None:
        region_ids = []
        for key in sorted(self._selected_face_region_sketch_keys):
            region_id = self._scene_region_id_from_key(key)
            if region_id in self._scene_project.regions:
                region_ids.append(region_id)
        region_ids.extend(sorted(self._scene_selection.active_region_ids))
        region_ids = sorted(set(region_ids))
        set_id = self._upsert_geometry_set_from_entities(
            entity_type="region",
            entity_ids=region_ids,
            preferred_name=self._ui("区域集合", "Region Set"),
        )
        if set_id is None:
            self._show_error(
                self._ui("集合创建失败", "Set capture failed"),
                self._ui("请先在画布选择至少一个区域。", "Please select at least one region first."),
            )
            return
        self._log(
            self._ui(
                f"已创建/更新区域集合 {set_id}，区域数={len(region_ids)}。",
                f"Captured region set {set_id} with {len(region_ids)} region(s).",
            )
        )

    def _capture_component_set_from_selected_component(self) -> None:
        component_id = self._component_manager_selected_id() or self._scene_selection.active_component_id
        if component_id is None or component_id not in self._scene_project.components:
            self._show_error(
                self._ui("集合创建失败", "Set capture failed"),
                self._ui("请先在 Assembly/Component 中选中一个组件。", "Please select a component in Assembly/Component first."),
            )
            return
        set_id = self._upsert_geometry_set_from_entities(
            entity_type="component",
            entity_ids=[component_id],
            preferred_name=self._ui("组件集合", "Component Set"),
        )
        if set_id is None:
            self._show_error(
                self._ui("集合创建失败", "Set capture failed"),
                self._ui("组件集合创建失败，请重试。", "Failed to capture component set."),
            )
            return
        self._log(
            self._ui(
                f"已创建/更新组件集合 {set_id}，组件={component_id}。",
                f"Captured component set {set_id} for {component_id}.",
            )
        )

    def _current_load_bc_target_mode(self) -> str:
        if not hasattr(self, "_load_bc_target_mode_combo"):
            return "mesh"
        data = self._load_bc_target_mode_combo.currentData()
        if isinstance(data, str) and data:
            return data
        return "mesh"

    def _current_load_bc_target_set_id(self) -> str | None:
        if not hasattr(self, "_load_bc_target_set_combo"):
            return None
        data = self._load_bc_target_set_combo.currentData()
        if isinstance(data, str) and data:
            return data
        return None

    def _current_load_bc_target_set(self) -> GeometrySetDef | None:
        set_id = self._current_load_bc_target_set_id()
        if not set_id:
            return None
        return self._scene_project.geometry_sets.get(set_id)

    def _is_geometry_set_used_by_load_bc_definition(self, set_id: str | None) -> bool:
        if not set_id:
            return False
        normalized = str(set_id)
        return any(
            definition.target_set_id == normalized
            for definition in self._scene_project.load_definitions.values()
        ) or any(
            definition.target_set_id == normalized
            for definition in self._scene_project.boundary_definitions.values()
        )

    def _preferred_geometry_set_id_for_pick(self, entity_type: str, *, allow_referenced: bool = False) -> str | None:
        candidates: list[str] = []
        current_set = self._current_load_bc_target_set()
        if current_set is not None and current_set.binding_mode == "geometry" and current_set.entity_type == entity_type:
            candidates.append(current_set.id)
        if self._last_geometry_pick_set_id:
            candidates.append(self._last_geometry_pick_set_id)
        for candidate_id in candidates:
            candidate = self._scene_project.geometry_sets.get(str(candidate_id))
            if candidate is None:
                continue
            if candidate.binding_mode != "geometry" or candidate.entity_type != entity_type:
                continue
            if not allow_referenced and self._is_geometry_set_used_by_load_bc_definition(candidate.id):
                continue
            return candidate.id
        return None

    def _on_load_bc_target_mode_changed(self, _: int) -> None:
        if self._pending_pick_context in {
            "geometry_point_target",
            "geometry_edge_target",
            "geometry_point_set_multi",
            "geometry_split_edge_point",
        }:
            self._pending_pick_context = None
            self._set_canvas_tool("select")
        mode = self._current_load_bc_target_mode()
        self._btn_load_bc_pick_geo_point.setEnabled(mode == "geometry")
        self._btn_load_bc_pick_geo_edge.setEnabled(mode == "geometry")
        if hasattr(self, "_btn_load_bc_pick_region"):
            self._btn_load_bc_pick_region.setEnabled(mode == "geometry")
        if hasattr(self, "_btn_load_bc_pick_geo_point_set"):
            self._btn_load_bc_pick_geo_point_set.setEnabled(mode == "geometry")
        if hasattr(self, "_btn_load_bc_split_edge_point"):
            self._btn_load_bc_split_edge_point.setEnabled(mode == "geometry")
        if hasattr(self, "_btn_load_bc_create_point_by_coord"):
            self._btn_load_bc_create_point_by_coord.setEnabled(mode == "geometry")
        if hasattr(self, "_btn_load_bc_create_edge_fraction_point"):
            self._btn_load_bc_create_edge_fraction_point.setEnabled(mode == "geometry")
        if hasattr(self, "_btn_load_bc_clear_point_set"):
            self._btn_load_bc_clear_point_set.setEnabled(mode == "geometry")
        if hasattr(self, "_btn_load_bc_finish_pick"):
            self._btn_load_bc_finish_pick.setEnabled(mode == "geometry")
        if hasattr(self, "_btn_load_bc_cleanup_points"):
            self._btn_load_bc_cleanup_points.setEnabled(mode == "geometry")
        self._load_point_node_spin.setEnabled(mode != "geometry")
        self._bc_point_node_spin.setEnabled(mode != "geometry")
        self._btn_pick_load_point.setText(
            self._ui("画布选几何点", "Pick Geometry Point") if mode == "geometry" else self._tr("rock.loadbc.pick_point_load", "画布选点")
        )
        self._btn_pick_bc_point.setText(
            self._ui("画布选几何点", "Pick Geometry Point") if mode == "geometry" else self._tr("rock.loadbc.pick_point_bc", "画布选点")
        )
        self._on_load_bc_target_set_changed(self._load_bc_target_set_combo.currentIndex())
        self._sync_load_bc_geometry_overlay()

    def _start_pick_geometry_point_target(self) -> None:
        if not self._scene_project.regions:
            self._show_error(
                self._ui("选择失败", "Pick failed"),
                self._ui("当前没有可用几何区域，请先在 Part 中绘制封闭草图。", "No geometric region available. Create closed sketch first."),
            )
            return
        if self._pending_pick_context == "geometry_point_target":
            self._pending_pick_context = None
            self._set_canvas_tool("select")
            return
        self._pending_pick_context = "geometry_point_target"
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("pick_geometry")
        self._sync_load_bc_geometry_overlay()
        self._log(self._ui("几何选点已激活：请在画布中点选几何点/边/区域内部点。", "Geometry point pick enabled. Click a geometric point/edge/interior point on canvas."))

    def _start_pick_geometry_edge_target(self) -> None:
        if not self._scene_project.geometry_edges:
            self._show_error(
                self._ui("选择失败", "Pick failed"),
                self._ui("当前没有可用几何边，请先在 Part 中绘制封闭草图。", "No geometric edge available. Create closed sketch first."),
            )
            return
        if self._pending_pick_context == "geometry_edge_target":
            self._pending_pick_context = None
            self._set_canvas_tool("select")
            return
        self._pending_pick_context = "geometry_edge_target"
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("pick_geometry")
        self._sync_load_bc_geometry_overlay()
        self._log(self._ui("几何选边已激活：请在画布中点选目标几何边。", "Geometry edge pick enabled. Click a target geometric edge on canvas."))

    def _start_pick_distributed_edge_target(self) -> None:
        if hasattr(self, "_load_bc_target_mode_combo"):
            idx = self._load_bc_target_mode_combo.findData("geometry")
            if idx >= 0 and self._load_bc_target_mode_combo.currentIndex() != idx:
                self._load_bc_target_mode_combo.setCurrentIndex(idx)
        edge_idx = self._bc_target_combo.findData("edge")
        if edge_idx >= 0:
            self._bc_target_combo.setCurrentIndex(edge_idx)
        self._start_pick_geometry_edge_target()
        if hasattr(self, "_load_bc_set_hint") and self._pending_pick_context == "geometry_edge_target":
            self._load_bc_set_hint.setText(
                self._ui(
                    "分布力目标需要边集：请在画布上点选一条几何边，选中后再点击“添加分布力”。",
                    "Distributed load needs an edge set: click a geometry edge on the canvas, then add the distributed load.",
                )
            )

    def _prepare_edge_pressure_creation(self) -> None:
        if hasattr(self, "_load_dist_type_combo"):
            idx = self._load_dist_type_combo.findData("pressure")
            if idx >= 0:
                self._load_dist_type_combo.setCurrentIndex(idx)
        if hasattr(self, "_load_dist_direction_mode_combo"):
            idx = self._load_dist_direction_mode_combo.findData("normal")
            if idx >= 0:
                self._load_dist_direction_mode_combo.setCurrentIndex(idx)
        self._start_pick_distributed_edge_target()

    def _prepare_region_body_force_creation(self) -> None:
        self._region_load_vec_x.setValue(0.0)
        self._region_load_vec_y.setValue(-1.0)
        self._start_pick_load_bc_region_target()

    def _prepare_gravity_creation(self) -> None:
        self._region_load_vec_x.setValue(0.0)
        self._region_load_vec_y.setValue(-1.0)
        self._start_pick_load_bc_region_target()

    def _start_pick_load_bc_region_target(self) -> None:
        if hasattr(self, "_load_bc_target_mode_combo"):
            idx = self._load_bc_target_mode_combo.findData("geometry")
            if idx >= 0 and self._load_bc_target_mode_combo.currentIndex() != idx:
                self._load_bc_target_mode_combo.setCurrentIndex(idx)
        has_mesh_elements = bool(self._model is not None and self._model.mesh.elements)
        sketch_regions = self._collect_sketch_face_regions()
        if not has_mesh_elements and not sketch_regions:
            self._show_error(
                self._ui("选择失败", "Pick failed"),
                self._ui("当前没有可选区域。请先在 Part 中绘制封闭区域或生成网格。", "No selectable region is available. Create a closed sketch face or mesh first."),
            )
            return
        self._selected_face_region_element_ids.clear()
        self._selected_face_region_sketch_keys.clear()
        self._selected_element_ids.clear()
        self._scene_selection.active_region_ids.clear()
        self._mesh_canvas.set_highlighted_elements(set())
        self._mesh_canvas.set_highlighted_sketch_faces([])
        self._pending_pick_context = "load_bc_region" if has_mesh_elements else "load_bc_region_sketch"
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("select")
        self._sync_pickable_sketch_faces_to_canvas()
        self._load_bc_set_hint.setText(
            self._ui(
                "区域选择已开启：在画布中连续点选区域，随后点击“由选区创建区域集合”。",
                "Region pick active: select regions on the canvas, then capture a region set.",
            )
        )
        self._log(
            self._ui(
                "Load/BC 区域拾取已激活：可用于区域体力/自重。Esc 结束。",
                "Load/BC region picking enabled for body force/gravity. Press Esc to finish.",
            )
        )

    def _toggle_pick_geometry_point_set_multi(self) -> None:
        if not self._scene_project.regions:
            self._show_error(
                self._ui("选择失败", "Pick failed"),
                self._ui("当前没有可用几何区域，请先在 Part 中绘制封闭草图。", "No geometric region available. Create closed sketch first."),
            )
            return
        if self._pending_pick_context == "geometry_point_set_multi":
            self._pending_pick_context = None
            self._set_canvas_tool("select")
            self._load_bc_set_hint.setText(
                self._ui("已退出连续点集拾取。", "Continuous point-set pick disabled.")
            )
            return
        current_set = self._current_load_bc_target_set()
        if (
            current_set is not None
            and current_set.binding_mode == "geometry"
            and current_set.entity_type == "point"
            and not self._is_geometry_set_used_by_load_bc_definition(current_set.id)
        ):
            self._selected_geometry_point_ids = {str(item) for item in current_set.entity_ids}
        self._pending_pick_context = "geometry_point_set_multi"
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("pick_geometry")
        self._load_bc_set_hint.setText(
            self._ui(
                f"连续点集拾取已开启：点击可增删点，当前已选 {len(self._selected_geometry_point_ids)} 个，Esc 结束。",
                f"Continuous point-set pick active: click to add/remove points, selected={len(self._selected_geometry_point_ids)} (Esc to finish).",
            )
        )
        self._sync_load_bc_geometry_overlay()
        self._log(
            self._ui(
                "连续点集拾取已激活：可连续点选多个几何点，系统会实时更新点集。",
                "Continuous point-set pick enabled: select multiple geometry points and update set in real time.",
            )
        )

    def _toggle_pick_split_edge_point(self) -> None:
        if not self._scene_project.geometry_edges:
            self._show_error(
                self._ui("选择失败", "Pick failed"),
                self._ui("当前没有可用几何边，请先在 Part 中绘制封闭草图。", "No geometric edge available. Create closed sketch first."),
            )
            return
        if self._pending_pick_context == "geometry_split_edge_point":
            self._pending_pick_context = None
            self._set_canvas_tool("select")
            self._load_bc_set_hint.setText(
                self._ui("已退出边上插点模式。", "Edge split-point mode disabled.")
            )
            return
        current_set = self._current_load_bc_target_set()
        if (
            current_set is not None
            and current_set.binding_mode == "geometry"
            and current_set.entity_type == "point"
            and not self._is_geometry_set_used_by_load_bc_definition(current_set.id)
        ):
            self._selected_geometry_point_ids = {str(item) for item in current_set.entity_ids}
        self._pending_pick_context = "geometry_split_edge_point"
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("pick_geometry")
        self._load_bc_set_hint.setText(
            self._ui(
                f"边上插点模式已开启：点击边可插入分段点，当前点集 {len(self._selected_geometry_point_ids)} 个，Esc 结束。",
                f"Edge split-point mode active: click edges to insert split points, selected={len(self._selected_geometry_point_ids)} (Esc to finish).",
            )
        )
        self._sync_load_bc_geometry_overlay()
        self._log(
            self._ui(
                "边上插点已激活：点击几何边会创建边上点并自动加入点集。",
                "Edge split-point mode enabled: clicking an edge creates an edge point and adds it to the point set.",
            )
        )

    def _clear_geometry_point_set_selection(self) -> None:
        self._selected_geometry_point_ids.clear()
        self._sync_load_bc_geometry_overlay()
        self._load_bc_set_hint.setText(
            self._ui("已清空当前点集选择。", "Current point-set selection cleared.")
        )

    def _finish_load_bc_geometry_picking(self) -> None:
        if self._pending_pick_context not in {
            "geometry_point_target",
            "geometry_edge_target",
            "geometry_point_set_multi",
            "geometry_split_edge_point",
        }:
            return
        self._pending_pick_context = None
        self._set_canvas_tool("select")
        self._sync_load_bc_geometry_overlay()
        self._load_bc_set_hint.setText(
            self._ui("已完成几何拾取。", "Geometry picking finished.")
        )
        self._log(
            self._ui(
                "几何拾取已完成，可继续输入参数并添加载荷/约束。",
                "Geometry picking finished. You can now set parameters and add load/BC.",
            )
        )

    def _cleanup_unused_manual_geometry_points(self) -> None:
        protected_point_ids: set[str] = set(self._selected_geometry_point_ids)
        current_set = self._current_load_bc_target_set()
        if current_set is not None and current_set.binding_mode == "geometry" and current_set.entity_type == "point":
            protected_point_ids.update(str(item) for item in current_set.entity_ids)
        for geometry_set in self._scene_project.geometry_sets.values():
            if geometry_set.binding_mode != "geometry" or geometry_set.entity_type != "point":
                continue
            protected_point_ids.update(str(item) for item in geometry_set.entity_ids)

        removed_ids: list[str] = []
        for point_id, point in list(self._scene_project.geometry_points.items()):
            if point.source != "manual":
                continue
            if str(point_id) in protected_point_ids:
                continue
            removed_ids.append(str(point_id))
            self._scene_project.geometry_points.pop(point_id, None)

        if not removed_ids:
            self._load_bc_set_hint.setText(
                self._ui("没有可清理的辅助点。", "No removable helper points found.")
            )
            return

        removed_set_links = 0
        for geometry_set in self._scene_project.geometry_sets.values():
            if geometry_set.binding_mode != "geometry" or geometry_set.entity_type != "point":
                continue
            before = len(geometry_set.entity_ids)
            geometry_set.entity_ids = [item for item in geometry_set.entity_ids if str(item) not in removed_ids]
            removed_set_links += max(0, before - len(geometry_set.entity_ids))

        self._selected_geometry_point_ids = {
            point_id for point_id in self._selected_geometry_point_ids if point_id in self._scene_project.geometry_points
        }
        self._rebuild_geometry_pick_cache()
        self._rebuild_scene_mesh_state()
        self._refresh_load_bc_target_set_combo(self._current_load_bc_target_set_id())
        self._sync_load_bc_geometry_overlay()
        self._load_bc_set_hint.setText(
            self._ui(
                f"已清理 {len(removed_ids)} 个未使用辅助点（更新集合引用 {removed_set_links} 处）。",
                f"Removed {len(removed_ids)} unused helper point(s) (updated {removed_set_links} set link(s)).",
            )
        )
        self._log(
            self._ui(
                f"Load/BC 清理完成：删除未使用辅助点 {len(removed_ids)} 个。",
                f"Load/BC cleanup complete: removed {len(removed_ids)} unused helper point(s).",
            )
        )

    def _commit_geometry_point_selection_to_set(self, *, point_id: str, message: str) -> None:
        self._selected_geometry_point_ids.add(str(point_id))
        preferred_set_id = self._preferred_geometry_set_id_for_pick("point")
        set_id = self._upsert_geometry_set_from_geometry_ids(
            entity_type="point",
            geometry_ids=sorted(self._selected_geometry_point_ids),
            preferred_name=self._ui("几何点集合", "Geometry Point Set"),
            preferred_set_id=preferred_set_id,
        )
        if set_id is not None:
            self._last_geometry_pick_set_id = set_id
            idx = self._load_bc_target_set_combo.findData(set_id)
            if idx >= 0:
                self._load_bc_target_set_combo.setCurrentIndex(idx)
        self._sync_load_bc_geometry_overlay()
        self._load_bc_set_hint.setText(message)

    def _create_geometry_point_by_coordinate_dialog(self) -> None:
        if self._current_load_bc_target_mode() != "geometry":
            self._load_bc_set_hint.setText(
                self._ui("请先切换到几何目标模式。", "Switch to Geometry Target mode first.")
            )
            return
        text, ok = QInputDialog.getText(
            self,
            self._ui("按坐标创建点", "Create Point by Coordinate"),
            self._ui("输入坐标 x,y：", "Enter coordinate x,y:"),
            text="0.0, 0.0",
        )
        if not ok:
            return
        tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
        if len(tokens) != 2:
            self._show_error(
                self._ui("创建点失败", "Create point failed"),
                self._ui("坐标格式应为 x,y。", "Coordinate format must be x,y."),
            )
            return
        try:
            x = float(tokens[0])
            y = float(tokens[1])
        except ValueError:
            self._show_error(
                self._ui("创建点失败", "Create point failed"),
                self._ui("坐标必须是数字。", "Coordinate values must be numeric."),
            )
            return
        point_id = self._resolve_or_create_geometry_point_at(x, y)
        if point_id is None:
            self._show_error(
                self._ui("创建点失败", "Create point failed"),
                self._ui("该坐标不在当前封闭几何区域内，也未命中边界。", "The coordinate is not inside the current closed geometry and did not hit an edge."),
            )
            return
        self._commit_geometry_point_selection_to_set(
            point_id=point_id,
            message=self._ui(
                f"已按坐标创建/选择点 {point_id}，并加入当前点集。",
                f"Created/selected point {point_id} by coordinate and added it to the current point set.",
            ),
        )

    def _create_geometry_point_on_selected_edge_fraction_dialog(self) -> None:
        if self._current_load_bc_target_mode() != "geometry":
            self._load_bc_set_hint.setText(
                self._ui("请先切换到几何目标模式。", "Switch to Geometry Target mode first.")
            )
            return
        edge_ids = sorted(self._selected_geometry_edge_ids)
        current_set = self._current_load_bc_target_set()
        if not edge_ids and current_set is not None and current_set.binding_mode == "geometry" and current_set.entity_type == "edge":
            edge_ids = [str(item) for item in current_set.entity_ids]
        if not edge_ids:
            self._start_pick_geometry_edge_target()
            self._load_bc_set_hint.setText(
                self._ui("请先选择一条几何边，再点击“边上比例点”。", "Pick an edge first, then click Edge Fraction Point.")
            )
            return
        ratio, ok = QInputDialog.getDouble(
            self,
            self._ui("边上比例点", "Edge Fraction Point"),
            self._ui("输入边参数 t，0=起点，1=终点：", "Enter edge parameter t, 0=start, 1=end:"),
            0.5,
            0.0,
            1.0,
            6,
        )
        if not ok:
            return
        edge_id = edge_ids[0]
        edge = self._scene_project.geometry_edges.get(edge_id)
        if edge is None:
            return
        start = self._scene_project.geometry_points.get(edge.start_point_id)
        end = self._scene_project.geometry_points.get(edge.end_point_id)
        if start is None or end is None:
            return
        t = max(0.0, min(1.0, float(ratio)))
        if t <= 1e-5:
            point_id = edge.start_point_id
        elif t >= 1.0 - 1e-5:
            point_id = edge.end_point_id
        else:
            point_id = None
            for existing_id, point in self._scene_project.geometry_points.items():
                if point.owner_edge_id == edge_id and point.role == "edge_point" and point.param is not None and abs(float(point.param) - t) <= 1e-4:
                    point_id = existing_id
                    break
            if point_id is None:
                px, py = self._interpolate_segment((float(start.x), float(start.y)), (float(end.x), float(end.y)), t)
                point_id = self._next_manual_geometry_point_id()
                self._scene_project.geometry_points[point_id] = GeometryPointDef(
                    id=point_id,
                    name=self._ui("边上比例点", "Edge Fraction Point"),
                    x=float(px),
                    y=float(py),
                    owner_loop_id=edge.loop_id,
                    owner_edge_id=edge_id,
                    param=t,
                    source="manual",
                    role="edge_point",
                )
                self._rebuild_geometry_pick_cache()
        self._commit_geometry_point_selection_to_set(
            point_id=point_id,
            message=self._ui(
                f"已在边 {edge_id} 的 t={t:.4f} 处创建/选择保点。",
                f"Created/selected retained point on edge {edge_id} at t={t:.4f}.",
            ),
        )

    def _on_load_bc_target_set_changed(self, _: int) -> None:
        if not hasattr(self, "_load_bc_target_set_combo") or not hasattr(self, "_load_bc_set_hint"):
            return
        self._sync_load_bc_geometry_overlay()
        set_id = self._load_bc_target_set_combo.currentData()
        if not isinstance(set_id, str) or not set_id:
            mode = self._current_load_bc_target_mode()
            self._load_bc_set_hint.setText(
                self._ui(
                    "未选择集合：将按当前参数施加载荷与约束。"
                    if mode == "mesh"
                    else "未选择几何集合：先点“几何选点/几何选边”，再添加载荷或约束。",
                    "No set selected: loads/BC will use current controls."
                    if mode == "mesh"
                    else "No geometry set selected: use geometry pick first, then apply load/BC.",
                )
            )
            return
        item = self._scene_project.geometry_sets.get(set_id)
        if item is None:
            self._load_bc_set_hint.setText(
                self._ui("集合不存在，请重新选择。", "Selected set is unavailable; please choose again.")
            )
            return
        resolved_node_ids = self._scene_mesh_state.set_to_node_ids.get(set_id, [])
        resolved_nodes = len(resolved_node_ids)
        mode = self._current_load_bc_target_mode()
        if mode == "geometry":
            if resolved_nodes > 0:
                if item.entity_type == "point" and resolved_nodes == 1:
                    node_id = int(resolved_node_ids[0])
                    self._load_point_node_spin.setValue(node_id)
                    self._bc_point_node_spin.setValue(node_id)
                self._load_bc_set_hint.setText(
                    self._ui(
                        f"已就绪：{item.name} 已解析为 {resolved_nodes} 个网格节点，可直接施加载荷/约束。",
                        f"Ready: {item.name} resolved to {resolved_nodes} mesh node(s).",
                    )
                )
            else:
                self._load_bc_set_hint.setText(
                    self._ui(
                        f"未就绪：{item.name} 尚未解析到网格节点。点击“添加集中力/分布力/应用约束”会自动尝试保点重网格。",
                        f"Not ready: {item.name} is unresolved. Applying load/BC will auto-attempt geometry-preserving remesh.",
                    )
                )
            return
        status_text = self._ui("resolved" if resolved_nodes > 0 else "unresolved", "resolved" if resolved_nodes > 0 else "unresolved")
        self._load_bc_set_hint.setText(
            self._ui(
                f"当前集合: {item.name} [{item.entity_type}/{item.binding_mode}]，实体数={len(item.entity_ids)}，解析节点={resolved_nodes} ({status_text})。",
                f"Current set: {item.name} [{item.entity_type}/{item.binding_mode}], entities={len(item.entity_ids)}, resolved_nodes={resolved_nodes} ({status_text}).",
            )
        )

    def _auto_resolve_selected_geometry_target_set(self, *, purpose: str) -> bool:
        if self._current_load_bc_target_mode() != "geometry":
            return True
        set_id = self._current_load_bc_target_set_id()
        if not set_id:
            return True
        target_set = self._scene_project.geometry_sets.get(set_id)
        if target_set is None:
            self._show_error(
                self._ui("目标集合异常", "Invalid target set"),
                self._ui("当前目标集合不存在，请重新选择。", "Selected target set is unavailable. Please choose again."),
            )
            return False
        if target_set.entity_type not in {"point", "edge"}:
            return True
        if self._scene_mesh_state.set_to_node_ids.get(set_id, []):
            return True
        if not self._scene_project.loops:
            self._show_error(
                self._ui("解析失败", "Resolve failed"),
                self._ui(
                    "当前没有可用于保点重网格的封闭几何。请先在 Part 中绘制闭合轮廓。",
                    "No closed sketch loops available for geometry-preserving remesh.",
                ),
            )
            return False

        self._log(
            self._ui(
                f"{purpose}前检测到几何目标未解析，正在自动执行保点重网格...",
                f"Geometry target unresolved before {purpose}; auto-running geometry-preserving remesh...",
            )
        )
        try:
            model = self._build_model_from_scene_constrained()
        except Exception as exc:
            self._show_error(
                self._ui("解析失败", "Resolve failed"),
                self._ui(
                    f"自动保点重网格失败：{exc}",
                    f"Auto geometry-preserving remesh failed: {exc}",
                ),
            )
            return False
        self._clear_mapping_state()
        self._set_model(model)
        mapped = self._apply_geometry_material_rules_to_mesh()
        if mapped > 0:
            self._set_model(self._model)
        self._rebuild_scene_mesh_state()
        idx = self._load_bc_target_set_combo.findData(set_id)
        if idx >= 0:
            self._load_bc_target_set_combo.setCurrentIndex(idx)
        resolved_node_ids = self._scene_mesh_state.set_to_node_ids.get(set_id, [])
        if not resolved_node_ids:
            self._show_error(
                self._ui("解析失败", "Resolve failed"),
                self._ui(
                    "自动重网格后仍未解析到节点。请检查选点是否在封闭区域内、是否命中边界。",
                    "Target is still unresolved after remesh. Ensure picked point is inside region or edge hit is valid.",
                ),
            )
            return False
        self._log(
            self._ui(
                f"几何目标已自动解析成功：集合 {target_set.name} -> 节点 {len(resolved_node_ids)} 个。",
                f"Geometry target resolved automatically: set {target_set.name} -> {len(resolved_node_ids)} node(s).",
            )
        )
        return True

    def _next_load_definition_id(self) -> str:
        max_idx = 0
        for load_id in self._scene_project.load_definitions:
            token = str(load_id).rsplit(":", 1)[-1]
            if token.isdigit():
                max_idx = max(max_idx, int(token))
        return f"load:def:{max_idx + 1}"

    def _next_boundary_definition_id(self) -> str:
        max_idx = 0
        for bc_id in self._scene_project.boundary_definitions:
            token = str(bc_id).rsplit(":", 1)[-1]
            if token.isdigit():
                max_idx = max(max_idx, int(token))
        return f"bc:def:{max_idx + 1}"

    @staticmethod
    def _load_instances_match(left: Load, right: Load) -> bool:
        return (
            int(left.node_id) == int(right.node_id)
            and str(left.dof) == str(right.dof)
            and abs(float(left.value) - float(right.value)) <= 1e-10 * max(1.0, abs(float(left.value)), abs(float(right.value)))
        )

    @staticmethod
    def _bc_instances_match(left: BoundaryCondition, right: BoundaryCondition) -> bool:
        return (
            int(left.node_id) == int(right.node_id)
            and str(left.dof) == str(right.dof)
            and abs(float(left.value) - float(right.value)) <= 1e-12 * max(1.0, abs(float(left.value)), abs(float(right.value)))
        )

    def _remove_previous_scene_definition_expansion(self) -> None:
        if self._model is None:
            return
        remaining_loads = list(self._model.loads)
        for generated in self._resolved_scene_loads_by_def_id.values():
            for old_load in generated:
                for idx, item in enumerate(remaining_loads):
                    if self._load_instances_match(item, old_load):
                        del remaining_loads[idx]
                        break
        self._model.loads = remaining_loads

        remaining_bcs = list(self._model.boundary_conditions)
        for generated in self._resolved_scene_bcs_by_def_id.values():
            for old_bc in generated:
                for idx, item in enumerate(remaining_bcs):
                    if self._bc_instances_match(item, old_bc):
                        del remaining_bcs[idx]
                        break
        self._model.boundary_conditions = remaining_bcs
        self._resolved_scene_loads_by_def_id = {}
        self._resolved_scene_bcs_by_def_id = {}

    def _edge_node_sequences_for_set(self, target_set: GeometrySetDef) -> list[list[int]]:
        if self._model is None:
            return []
        node_by_id = {int(node.id): node for node in self._model.mesh.nodes}
        sequences: list[list[int]] = []
        if target_set.binding_mode == "geometry":
            for edge_id in target_set.entity_ids:
                sequence = [
                    int(node_id)
                    for node_id in self._scene_mesh_state.edge_to_node_ids.get(str(edge_id), [])
                    if int(node_id) in node_by_id
                ]
                if len(sequence) >= 2:
                    sequences.append(sequence)
            return sequences

        ids = []
        for node_id in self._scene_mesh_state.set_to_node_ids.get(target_set.id, []):
            node = node_by_id.get(int(node_id))
            if node is not None:
                ids.append(int(node_id))
        if len(ids) < 2:
            return []
        x_span = max(float(node_by_id[node_id].x) for node_id in ids) - min(float(node_by_id[node_id].x) for node_id in ids)
        y_span = max(float(node_by_id[node_id].y) for node_id in ids) - min(float(node_by_id[node_id].y) for node_id in ids)
        if x_span >= y_span:
            ids.sort(key=lambda node_id: (float(node_by_id[node_id].x), float(node_by_id[node_id].y), node_id))
        else:
            ids.sort(key=lambda node_id: (float(node_by_id[node_id].y), float(node_by_id[node_id].x), node_id))
        return [ids]

    def _distributed_loads_from_edge_sequences(
        self,
        definition: LoadDefinition,
        sequences: list[list[int]],
    ) -> list[Load]:
        if self._model is None:
            return []
        node_by_id = {int(node.id): node for node in self._model.mesh.nodes}
        vx = float(definition.vector_x)
        vy = float(definition.vector_y)
        norm = hypot(vx, vy)
        if norm <= 1e-12:
            return []
        dir_x = vx / norm
        dir_y = vy / norm

        segments: list[tuple[int, int, float, float, float]] = []
        cumulative = 0.0
        for sequence in sequences:
            for start_id, end_id in zip(sequence, sequence[1:]):
                start = node_by_id.get(int(start_id))
                end = node_by_id.get(int(end_id))
                if start is None or end is None or int(start_id) == int(end_id):
                    continue
                length = hypot(float(end.x) - float(start.x), float(end.y) - float(start.y))
                if length <= 1e-12:
                    continue
                segments.append((int(start_id), int(end_id), length, cumulative, cumulative + length))
                cumulative += length
        if cumulative <= 1e-12:
            return []

        coeff_by_node: dict[int, float] = {}
        denom = 0.0
        if definition.profile == "linear":
            for start_id, end_id, length, s0, s1 in segments:
                w0 = 0.2 + s0 / cumulative
                w1 = 0.2 + s1 / cumulative
                coeff_by_node[start_id] = coeff_by_node.get(start_id, 0.0) + length * (2.0 * w0 + w1) / 6.0
                coeff_by_node[end_id] = coeff_by_node.get(end_id, 0.0) + length * (w0 + 2.0 * w1) / 6.0
                denom += length * (w0 + w1) * 0.5
        else:
            for start_id, end_id, length, _, _ in segments:
                coeff_by_node[start_id] = coeff_by_node.get(start_id, 0.0) + 0.5 * length
                coeff_by_node[end_id] = coeff_by_node.get(end_id, 0.0) + 0.5 * length
            denom = cumulative
        if denom <= 1e-12:
            return []

        total_fx = float(definition.magnitude) * dir_x
        total_fy = float(definition.magnitude) * dir_y
        loads: list[Load] = []
        for node_id, coeff in sorted(coeff_by_node.items()):
            ratio = coeff / denom
            fx = total_fx * ratio
            fy = total_fy * ratio
            if abs(fx) > 1e-12:
                loads.append(Load(node_id=int(node_id), dof="fx", value=fx))
            if abs(fy) > 1e-12:
                loads.append(Load(node_id=int(node_id), dof="fy", value=fy))
        return loads

    def _pressure_loads_from_edge_sequences(
        self,
        definition: LoadDefinition,
        sequences: list[list[int]],
    ) -> list[Load]:
        if self._model is None:
            return []
        node_by_id = {int(node.id): node for node in self._model.mesh.nodes}
        coeff_by_node: dict[tuple[int, str], float] = {}
        mode = str(getattr(definition, "direction_mode", "normal") or "normal")
        vx = float(definition.vector_x)
        vy = float(definition.vector_y)
        norm = hypot(vx, vy)
        use_vector = mode == "vector" and norm > 1e-12
        if mode == "vector" and not use_vector:
            return []
        ux = vx / norm if use_vector else 0.0
        uy = vy / norm if use_vector else 0.0
        pressure = float(definition.magnitude)
        for sequence in sequences:
            for start_id, end_id in zip(sequence, sequence[1:]):
                start = node_by_id.get(int(start_id))
                end = node_by_id.get(int(end_id))
                if start is None or end is None or int(start_id) == int(end_id):
                    continue
                dx = float(end.x) - float(start.x)
                dy = float(end.y) - float(start.y)
                length = hypot(dx, dy)
                if length <= 1e-12:
                    continue
                if use_vector:
                    dir_x, dir_y = ux, uy
                else:
                    sign = -1.0 if mode == "reverse_normal" else 1.0
                    dir_x = sign * dy / length
                    dir_y = sign * -dx / length
                fx = pressure * dir_x * length * 0.5
                fy = pressure * dir_y * length * 0.5
                coeff_by_node[(int(start_id), "fx")] = coeff_by_node.get((int(start_id), "fx"), 0.0) + fx
                coeff_by_node[(int(end_id), "fx")] = coeff_by_node.get((int(end_id), "fx"), 0.0) + fx
                coeff_by_node[(int(start_id), "fy")] = coeff_by_node.get((int(start_id), "fy"), 0.0) + fy
                coeff_by_node[(int(end_id), "fy")] = coeff_by_node.get((int(end_id), "fy"), 0.0) + fy
        return [
            Load(node_id=node_id, dof=dof, value=value)
            for (node_id, dof), value in sorted(coeff_by_node.items())
            if abs(float(value)) > 1e-12
        ]

    def _region_or_component_loads_from_elements(self, definition: LoadDefinition, element_ids: list[int]) -> list[Load]:
        if self._model is None:
            return []
        node_by_id = {int(node.id): node for node in self._model.mesh.nodes}
        element_by_id = {int(element.id): element for element in self._model.mesh.elements}
        vx = float(definition.vector_x)
        vy = float(definition.vector_y)
        if definition.load_type == "gravity":
            if abs(vx) <= 1e-12 and abs(vy) <= 1e-12:
                vx, vy = 0.0, -1.0
        norm = hypot(vx, vy)
        if norm <= 1e-12:
            return []
        ux = vx / norm
        uy = vy / norm
        intensity = float(definition.magnitude)
        coeff_by_node: dict[tuple[int, str], float] = {}
        for element_id in sorted({int(item) for item in element_ids}):
            element = element_by_id.get(int(element_id))
            if element is None or len(element.connectivity) < 3:
                continue
            n1 = node_by_id.get(int(element.connectivity[0]))
            n2 = node_by_id.get(int(element.connectivity[1]))
            n3 = node_by_id.get(int(element.connectivity[2]))
            if n1 is None or n2 is None or n3 is None:
                continue
            area = abs(
                (float(n2.x) - float(n1.x)) * (float(n3.y) - float(n1.y))
                - (float(n3.x) - float(n1.x)) * (float(n2.y) - float(n1.y))
            ) * 0.5
            if area <= 1e-12:
                continue
            fx_each = intensity * ux * area / 3.0
            fy_each = intensity * uy * area / 3.0
            for node_id in (int(n1.id), int(n2.id), int(n3.id)):
                coeff_by_node[(node_id, "fx")] = coeff_by_node.get((node_id, "fx"), 0.0) + fx_each
                coeff_by_node[(node_id, "fy")] = coeff_by_node.get((node_id, "fy"), 0.0) + fy_each
        return [
            Load(node_id=node_id, dof=dof, value=value)
            for (node_id, dof), value in sorted(coeff_by_node.items())
            if abs(float(value)) > 1e-12
        ]

    def _resolve_scene_load_bc_definitions_to_model(self, *, rebuild_state: bool = True) -> list[str]:
        if self._model is None:
            return []
        self._remove_previous_scene_definition_expansion()
        if rebuild_state:
            self._rebuild_scene_mesh_state()

        unresolved: list[str] = []
        node_ids = {int(node.id) for node in self._model.mesh.nodes}
        generated_loads_by_def: dict[str, list[Load]] = {}
        generated_bcs_by_def: dict[str, list[BoundaryCondition]] = {}

        for definition in self._scene_project.load_definitions.values():
            if not getattr(definition, "active", True):
                continue
            target_set = self._scene_project.geometry_sets.get(definition.target_set_id)
            if target_set is None:
                unresolved.append(f"{definition.name}: target set missing")
                continue
            if target_set.entity_type != definition.target_entity_type:
                unresolved.append(
                    self._ui(
                        f"{definition.name}: 目标类型不匹配，需要 {definition.target_entity_type} 集合。",
                        f"{definition.name}: target type mismatch; expected a {definition.target_entity_type} set.",
                    )
                )
                continue
            generated: list[Load] = []
            if definition.load_type == "concentrated":
                target_nodes = [
                    int(node_id)
                    for node_id in self._scene_mesh_state.set_to_node_ids.get(target_set.id, [])
                    if int(node_id) in node_ids
                ]
                if len(target_nodes) != 1:
                    unresolved.append(
                        self._ui(
                            f"{definition.name}: 集中力目标尚未解析为唯一节点。",
                            f"{definition.name}: concentrated-load target is not resolved to exactly one node.",
                        )
                    )
                    continue
                vx = float(definition.vector_x)
                vy = float(definition.vector_y)
                norm = hypot(vx, vy)
                if norm <= 1e-12:
                    unresolved.append(self._ui(f"{definition.name}: 方向向量为零。", f"{definition.name}: zero direction vector."))
                    continue
                fx = float(definition.magnitude) * vx / norm
                fy = float(definition.magnitude) * vy / norm
                if abs(fx) > 1e-12:
                    generated.append(Load(node_id=target_nodes[0], dof="fx", value=fx))
                if abs(fy) > 1e-12:
                    generated.append(Load(node_id=target_nodes[0], dof="fy", value=fy))
            elif definition.load_type == "distributed":
                sequences = self._edge_node_sequences_for_set(target_set)
                if not sequences:
                    unresolved.append(
                        self._ui(
                            f"{definition.name}: 分布力边集尚未解析为边界节点序列。",
                            f"{definition.name}: distributed-load edge set is not resolved to boundary node sequences.",
                        )
                    )
                    continue
                generated = self._distributed_loads_from_edge_sequences(definition, sequences)
                if not generated:
                    unresolved.append(
                        self._ui(
                            f"{definition.name}: 分布力未生成有效等效节点力。",
                            f"{definition.name}: distributed load produced no valid nodal loads.",
                        )
                    )
                    continue
            elif definition.load_type == "pressure":
                sequences = self._edge_node_sequences_for_set(target_set)
                if not sequences:
                    unresolved.append(
                        self._ui(
                            f"{definition.name}: 边压力目标尚未解析为边界节点序列。",
                            f"{definition.name}: edge-pressure target is not resolved to boundary node sequences.",
                        )
                    )
                    continue
                generated = self._pressure_loads_from_edge_sequences(definition, sequences)
                if not generated:
                    unresolved.append(
                        self._ui(
                            f"{definition.name}: 边压力未生成有效等效节点力。",
                            f"{definition.name}: edge pressure produced no valid nodal loads.",
                        )
                    )
                    continue
            elif definition.load_type in {"body", "gravity"}:
                if target_set.entity_type not in {"region", "component"}:
                    unresolved.append(
                        self._ui(
                            f"{definition.name}: 区域体力/自重需要区域集或组件集。",
                            f"{definition.name}: body force/gravity needs a region or component set.",
                        )
                    )
                    continue
                element_ids = [
                    int(element_id)
                    for element_id in self._scene_mesh_state.set_to_element_ids.get(target_set.id, [])
                    if int(element_id) > 0
                ]
                if not element_ids:
                    unresolved.append(
                        self._ui(
                            f"{definition.name}: 区域/组件目标尚未解析到单元。",
                            f"{definition.name}: region/component target is not resolved to elements.",
                        )
                    )
                    continue
                generated = self._region_or_component_loads_from_elements(definition, element_ids)
                if not generated:
                    unresolved.append(
                        self._ui(
                            f"{definition.name}: 区域体力/自重未生成有效等效节点力。",
                            f"{definition.name}: body force/gravity produced no valid nodal loads.",
                        )
                    )
                    continue
            else:
                unresolved.append(
                    self._ui(
                        f"{definition.name}: 未支持的载荷类型 {definition.load_type}。",
                        f"{definition.name}: unsupported load type {definition.load_type}.",
                    )
                )
                continue
            self._model.loads.extend(generated)
            generated_loads_by_def[definition.id] = generated

        bc_by_key: dict[tuple[int, str], BoundaryCondition] = {
            (int(bc.node_id), str(bc.dof)): bc
            for bc in self._model.boundary_conditions
        }
        for definition in self._scene_project.boundary_definitions.values():
            if not getattr(definition, "active", True):
                continue
            target_set = self._scene_project.geometry_sets.get(definition.target_set_id)
            if target_set is None:
                unresolved.append(f"{definition.name}: target set missing")
                continue
            if target_set.entity_type != definition.target_entity_type:
                unresolved.append(
                    self._ui(
                        f"{definition.name}: 约束目标类型不匹配，需要 {definition.target_entity_type} 集合。",
                        f"{definition.name}: boundary target type mismatch; expected a {definition.target_entity_type} set.",
                    )
                )
                continue
            target_nodes = [
                int(node_id)
                for node_id in self._scene_mesh_state.set_to_node_ids.get(target_set.id, [])
                if int(node_id) in node_ids
            ]
            if not target_nodes:
                unresolved.append(
                    self._ui(
                        f"{definition.name}: 约束目标尚未解析到网格节点。",
                        f"{definition.name}: boundary target is not resolved to mesh nodes.",
                    )
                )
                continue
            dofs = ("ux",) if definition.direction == "x" else ("uy",) if definition.direction == "y" else ("ux", "uy")
            generated_bcs: list[BoundaryCondition] = []
            for node_id in target_nodes:
                for dof in dofs:
                    bc = BoundaryCondition(node_id=int(node_id), dof=dof, value=float(definition.value))
                    bc_by_key[(int(node_id), dof)] = bc
                    generated_bcs.append(bc)
            generated_bcs_by_def[definition.id] = generated_bcs

        self._model.boundary_conditions = sorted(
            bc_by_key.values(),
            key=lambda item: (int(item.node_id), str(item.dof)),
        )
        self._resolved_scene_loads_by_def_id = generated_loads_by_def
        self._resolved_scene_bcs_by_def_id = generated_bcs_by_def
        self._sync_load_bc_geometry_overlay()
        return unresolved

    def _sync_load_bc_geometry_overlay(self) -> None:
        if not hasattr(self, "_mesh_canvas"):
            return
        show_all_geometry = self._current_load_bc_target_mode() == "geometry" or bool(self._scene_project.load_definitions or self._scene_project.boundary_definitions)
        selected_point_ids = set(self._selected_geometry_point_ids)
        selected_edge_ids = set(self._selected_geometry_edge_ids)
        set_id = self._current_load_bc_target_set_id()
        if set_id:
            target_set = self._scene_project.geometry_sets.get(set_id)
            if target_set is not None and target_set.binding_mode == "geometry":
                if target_set.entity_type == "point":
                    selected_point_ids.update(str(item) for item in target_set.entity_ids)
                elif target_set.entity_type == "edge":
                    selected_edge_ids.update(str(item) for item in target_set.entity_ids)

        previews: list[dict[str, object]] = []
        for definition in self._scene_project.load_definitions.values():
            target_set = self._scene_project.geometry_sets.get(definition.target_set_id)
            if target_set is None or target_set.binding_mode != "geometry":
                continue
            if definition.load_type == "concentrated":
                for point_id in target_set.entity_ids:
                    point = self._scene_project.geometry_points.get(str(point_id))
                    if point is None:
                        continue
                    selected_point_ids.add(str(point_id))
                    previews.append(
                        {
                            "kind": "point",
                            "x": float(point.x),
                            "y": float(point.y),
                            "vx": float(definition.vector_x),
                            "vy": float(definition.vector_y),
                            "label": definition.name,
                            "active": bool(getattr(definition, "active", True)),
                        }
                    )
            elif definition.load_type in {"distributed", "pressure"}:
                for edge_id in target_set.entity_ids:
                    edge = self._scene_project.geometry_edges.get(str(edge_id))
                    if edge is None:
                        continue
                    start = self._scene_project.geometry_points.get(edge.start_point_id)
                    end = self._scene_project.geometry_points.get(edge.end_point_id)
                    if start is None or end is None:
                        continue
                    selected_edge_ids.add(str(edge_id))
                    previews.append(
                        {
                            "kind": "edge",
                            "x0": float(start.x),
                            "y0": float(start.y),
                            "x1": float(end.x),
                            "y1": float(end.y),
                            "vx": float(definition.vector_x),
                            "vy": float(definition.vector_y),
                            "label": definition.name,
                            "load_type": definition.load_type,
                            "direction_mode": getattr(definition, "direction_mode", "vector"),
                            "active": bool(getattr(definition, "active", True)),
                        }
                    )
            elif definition.load_type in {"body", "gravity"}:
                region_ids: list[str] = []
                if target_set.entity_type == "region":
                    region_ids = [str(item) for item in target_set.entity_ids]
                elif target_set.entity_type == "component":
                    for component_id in target_set.entity_ids:
                        component = self._scene_project.components.get(str(component_id))
                        if component is not None:
                            region_ids.extend(str(region_id) for region_id in component.region_ids)
                for region_id in sorted(set(region_ids)):
                    region = self._scene_project.regions.get(region_id)
                    if region is None:
                        continue
                    points_region = self._scene_region_points(region)
                    if len(points_region) < 3:
                        continue
                    cx, cy = region.centroid if region.centroid != (0.0, 0.0) else polygon_centroid(points_region)
                    previews.append(
                        {
                            "kind": "region",
                            "points": points_region,
                            "x": float(cx),
                            "y": float(cy),
                            "vx": float(definition.vector_x),
                            "vy": float(definition.vector_y),
                            "label": definition.name,
                            "load_type": definition.load_type,
                            "active": bool(getattr(definition, "active", True)),
                        }
                    )

        bc_previews: list[dict[str, object]] = []
        for definition in self._scene_project.boundary_definitions.values():
            target_set = self._scene_project.geometry_sets.get(definition.target_set_id)
            if target_set is None or target_set.binding_mode != "geometry":
                continue
            if target_set.entity_type == "point":
                for point_id in target_set.entity_ids:
                    point = self._scene_project.geometry_points.get(str(point_id))
                    if point is None:
                        continue
                    selected_point_ids.add(str(point_id))
                    bc_previews.append(
                        {
                            "kind": "point",
                            "x": float(point.x),
                            "y": float(point.y),
                            "direction": definition.direction,
                            "label": definition.name,
                            "active": bool(getattr(definition, "active", True)),
                        }
                    )
            elif target_set.entity_type == "edge":
                for edge_id in target_set.entity_ids:
                    edge = self._scene_project.geometry_edges.get(str(edge_id))
                    if edge is None:
                        continue
                    start = self._scene_project.geometry_points.get(edge.start_point_id)
                    end = self._scene_project.geometry_points.get(edge.end_point_id)
                    if start is None or end is None:
                        continue
                    selected_edge_ids.add(str(edge_id))
                    bc_previews.append(
                        {
                            "kind": "edge",
                            "x0": float(start.x),
                            "y0": float(start.y),
                            "x1": float(end.x),
                            "y1": float(end.y),
                            "direction": definition.direction,
                            "label": definition.name,
                            "active": bool(getattr(definition, "active", True)),
                        }
                    )

        points: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []
        if show_all_geometry:
            required_point_ids = set(selected_point_ids)
            for point_id, point in self._scene_project.geometry_points.items():
                if point.source == "manual" or point.role != "vertex":
                    required_point_ids.add(str(point_id))
            visible_point_ids = required_point_ids
            for point_id in sorted(visible_point_ids):
                point = self._scene_project.geometry_points.get(point_id)
                if point is None:
                    continue
                points.append(
                    {
                        "id": point_id,
                        "x": float(point.x),
                        "y": float(point.y),
                        "label": point.name or point_id.rsplit(":", 1)[-1],
                    }
                )
            for edge_id, edge in sorted(self._scene_project.geometry_edges.items()):
                start = self._scene_project.geometry_points.get(edge.start_point_id)
                end = self._scene_project.geometry_points.get(edge.end_point_id)
                if start is None or end is None:
                    continue
                edges.append(
                    {
                        "id": edge_id,
                        "x0": float(start.x),
                        "y0": float(start.y),
                        "x1": float(end.x),
                        "y1": float(end.y),
                        "label": edge.name or edge_id.rsplit(":", 1)[-1],
                    }
                )

        self._mesh_canvas.set_load_bc_geometry_overlay(points, edges, selected_point_ids, selected_edge_ids, previews, bc_previews)

    def _current_results_scope_type(self) -> str:
        if not hasattr(self, "_results_scope_type_combo"):
            return str(self._results_scope_type)
        data = self._results_scope_type_combo.currentData()
        if isinstance(data, str) and data:
            return data
        return "all"

    def _refresh_results_scope_targets(self, preferred_target_id: str | None = None) -> None:
        if not hasattr(self, "_results_scope_target_combo"):
            return
        scope_type = self._current_results_scope_type()
        combo = self._results_scope_target_combo
        current_data = combo.currentData()
        current_target = preferred_target_id
        if current_target is None and isinstance(current_data, str):
            current_target = current_data
        combo.blockSignals(True)
        combo.clear()
        combo.setEnabled(scope_type != "all")
        if scope_type == "all":
            combo.addItem(self._ui("全部", "All"), "")
        elif scope_type == "component":
            for component_id in sorted(self._scene_project.components):
                component = self._scene_project.components[component_id]
                elem_count = len(self._scene_mesh_state.component_to_element_ids.get(component_id, []))
                combo.addItem(f"{component.name} ({elem_count})", component_id)
        elif scope_type == "region":
            for region_id in sorted(self._scene_project.regions):
                region = self._scene_project.regions[region_id]
                elem_count = len(self._scene_mesh_state.region_to_element_ids.get(region_id, []))
                combo.addItem(f"{region.name} ({elem_count})", region_id)
        elif scope_type == "material":
            if self._model is not None:
                for material in sorted(self._model.materials, key=lambda item: int(item.id)):
                    count = 0
                    for element in self._model.mesh.elements:
                        if int(element.material_id) == int(material.id):
                            count += 1
                    combo.addItem(f"{self._material_display_name(material.id)} [{material.id}] ({count})", str(int(material.id)))
        elif scope_type == "set":
            for set_id in sorted(self._scene_project.geometry_sets):
                item = self._scene_project.geometry_sets[set_id]
                combo.addItem(f"{item.name} [{item.entity_type}/{item.binding_mode}:{len(item.entity_ids)}]", set_id)

        idx = combo.findData(current_target) if current_target else -1
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        selected_data = combo.currentData()
        self._results_scope_type = scope_type
        self._results_scope_target_id = str(selected_data) if isinstance(selected_data, str) else ""
        self._on_results_scope_target_changed(combo.currentIndex())

    def _clear_results_scope_filter(self) -> None:
        if not hasattr(self, "_results_scope_type_combo"):
            return
        idx = self._results_scope_type_combo.findData("all")
        self._results_scope_type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._results_scope_target_id = ""
        self._refresh_results_scope_targets("")
        self._apply_results_scope_filter()

    def _on_results_scope_type_changed(self, _: int) -> None:
        self._results_scope_type = self._current_results_scope_type()
        self._refresh_results_scope_targets("")

    def _on_results_scope_target_changed(self, _: int) -> None:
        if not hasattr(self, "_results_scope_target_combo"):
            return
        data = self._results_scope_target_combo.currentData()
        self._results_scope_target_id = str(data) if isinstance(data, str) else ""
        self._apply_results_scope_filter()

    def _results_scope_element_ids(self) -> set[int] | None:
        if self._model is None:
            return None
        scope_type = self._current_results_scope_type()
        if scope_type == "all":
            return None
        if not hasattr(self, "_results_scope_target_combo"):
            return None
        target_id = self._results_scope_target_combo.currentData()
        if not isinstance(target_id, str) or not target_id:
            return None
        if scope_type == "component":
            return set(int(item) for item in self._scene_mesh_state.component_to_element_ids.get(target_id, []))
        if scope_type == "region":
            return set(int(item) for item in self._scene_mesh_state.region_to_element_ids.get(target_id, []))
        if scope_type == "material":
            try:
                mid = int(target_id)
            except ValueError:
                return set()
            return {int(element.id) for element in self._model.mesh.elements if int(element.material_id) == mid}
        if scope_type == "set":
            element_ids = set(int(item) for item in self._scene_mesh_state.set_to_element_ids.get(target_id, []))
            if element_ids:
                return element_ids
            node_ids = set(int(item) for item in self._scene_mesh_state.set_to_node_ids.get(target_id, []))
            if not node_ids:
                return set()
            mapped: set[int] = set()
            for element in self._model.mesh.elements:
                if any(int(node_id) in node_ids for node_id in element.connectivity):
                    mapped.add(int(element.id))
            return mapped
        return None

    def _results_scope_node_ids(self) -> set[int] | None:
        if self._model is None:
            return None
        element_ids = self._results_scope_element_ids()
        if element_ids is None:
            return None
        node_ids: set[int] = set()
        for element in self._model.mesh.elements:
            if int(element.id) not in element_ids:
                continue
            for node_id in element.connectivity:
                node_ids.add(int(node_id))
        scope_type = self._current_results_scope_type()
        if scope_type == "set" and hasattr(self, "_results_scope_target_combo"):
            target_id = self._results_scope_target_combo.currentData()
            if isinstance(target_id, str) and target_id:
                node_ids.update(int(item) for item in self._scene_mesh_state.set_to_node_ids.get(target_id, []))
        return node_ids

    def _apply_results_scope_filter(self) -> None:
        if self._model is None:
            return
        visible_elements = self._results_scope_element_ids()
        visible_nodes = self._results_scope_node_ids()
        if visible_elements is not None:
            self._selected_element_ids = {eid for eid in self._selected_element_ids if eid in visible_elements}
        if visible_nodes is not None:
            self._selected_node_ids = {nid for nid in self._selected_node_ids if nid in visible_nodes}
        self._mesh_canvas.set_highlighted_elements(self._selected_element_ids)
        self._mesh_canvas.set_highlighted_nodes(self._selected_node_ids)
        self._apply_component_visibility_filter_to_canvas()
        self._refresh_node_table()
        self._refresh_element_table()
        self._update_visualization_metrics()

    def _collect_sketch_face_regions(self) -> list[dict[str, object]]:
        regions: list[dict[str, object]] = []
        for region in self._scene_project.regions.values():
            if region.source != "sketch":
                continue
            points = self._normalize_polygon_points(self._scene_region_points(region))
            if len(points) < 3:
                continue
            regions.append(
                {
                    "key": region.id.replace("region:", "", 1),
                    "points": points,
                    "label": region.name,
                    "area": abs(scene_polygon_area(points)),
                }
            )
        if regions:
            regions.sort(key=lambda item: (float(item.get("area", 0.0)), str(item.get("key", ""))))
            return regions
        return self._collect_raw_sketch_face_regions()

    def _rebuild_sketch_face_pick_cache(self) -> None:
        cache: list[dict[str, object]] = []
        for item in self._collect_sketch_face_regions():
            points_raw = item.get("points", [])
            if not isinstance(points_raw, list):
                continue
            points = [(float(px), float(py)) for px, py in points_raw]
            if len(points) < 3:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            cache.append(
                {
                    "key": str(item.get("key", "")),
                    "label": str(item.get("label", "")),
                    "area": float(item.get("area", abs(scene_polygon_area(points)))),
                    "points": points,
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                }
            )
        cache.sort(key=lambda entry: (float(entry.get("area", 0.0)), str(entry.get("key", ""))))
        self._sketch_face_pick_cache = cache

    def _rebuild_geometry_pick_cache(self) -> None:
        point_cache: list[tuple[str, float, float]] = []
        sketch_vertex_candidates: list[tuple[str, str, float, float, float]] = []
        vertex_valence: dict[str, int] = {}
        for edge in self._scene_project.geometry_edges.values():
            start_id = str(edge.start_point_id)
            end_id = str(edge.end_point_id)
            vertex_valence[start_id] = int(vertex_valence.get(start_id, 0)) + 1
            vertex_valence[end_id] = int(vertex_valence.get(end_id, 0)) + 1
        for point_id, point in self._scene_project.geometry_points.items():
            pid = str(point_id)
            px = float(point.x)
            py = float(point.y)
            if str(point.source) == "sketch" and str(point.role) == "vertex":
                sketch_vertex_candidates.append(
                    (
                        pid,
                        str(point.owner_loop_id or ""),
                        float(point.param if point.param is not None else 0.0),
                        px,
                        py,
                    )
                )
                continue
            point_cache.append((pid, px, py))
        # Closed circles/arcs may be approximated by many sketch vertices.  They are
        # edge geometry, not user-facing load points; keep only true branching/end
        # vertices in the point cache and let endpoint clicks resolve through edges.
        for item in sorted(sketch_vertex_candidates, key=lambda value: (value[1], value[2], value[0])):
            point_id = str(item[0])
            if int(vertex_valence.get(point_id, 0)) != 2:
                point_cache.append((point_id, float(item[3]), float(item[4])))
        edge_cache: list[tuple[str, float, float, float, float, float, float, float, float]] = []
        for edge_id, edge in self._scene_project.geometry_edges.items():
            start = self._scene_project.geometry_points.get(edge.start_point_id)
            end = self._scene_project.geometry_points.get(edge.end_point_id)
            if start is None or end is None:
                continue
            x0 = float(start.x)
            y0 = float(start.y)
            x1 = float(end.x)
            y1 = float(end.y)
            edge_cache.append(
                (
                    str(edge_id),
                    x0,
                    y0,
                    x1,
                    y1,
                    min(x0, x1),
                    min(y0, y1),
                    max(x0, x1),
                    max(y0, y1),
                )
            )
        self._geometry_pick_point_cache = point_cache
        self._geometry_pick_edge_cache = edge_cache

    def _ensure_geometry_pick_cache(self) -> None:
        if len(self._geometry_pick_point_cache) != len(self._scene_project.geometry_points):
            self._rebuild_geometry_pick_cache()
            return
        if len(self._geometry_pick_edge_cache) != len(self._scene_project.geometry_edges):
            self._rebuild_geometry_pick_cache()

    def _update_sketch_face_highlight(self) -> None:
        if not self._selected_face_region_sketch_keys:
            self._mesh_canvas.set_highlighted_sketch_faces([])
            return
        if not self._sketch_face_pick_cache:
            self._rebuild_sketch_face_pick_cache()
        regions_by_key = {
            str(item["key"]): item
            for item in self._sketch_face_pick_cache
        }
        faces: list[list[tuple[float, float]]] = []
        for key in self._selected_face_region_sketch_keys:
            region = regions_by_key.get(key)
            if region is None:
                continue
            points = region.get("points", [])
            if isinstance(points, list):
                faces.append([(float(x), float(y)) for x, y in points])
        self._mesh_canvas.set_highlighted_sketch_faces(faces)

    def _sync_pickable_sketch_faces_to_canvas(self) -> None:
        if self._pending_pick_context not in {"material_face_sketch", "component_face_sketch", "mesh_region_sketch", "load_bc_region_sketch"}:
            self._mesh_canvas.set_pickable_sketch_faces([])
            return
        faces: list[list[tuple[float, float]]] = []
        for item in self._collect_sketch_face_regions():
            points_raw = item.get("points", [])
            points = self._normalize_polygon_points(points_raw if isinstance(points_raw, list) else [])
            if len(points) >= 3:
                faces.append(points)
        self._mesh_canvas.set_pickable_sketch_faces(faces)

    def _sync_material_sketch_faces_to_canvas(self) -> None:
        faces: list[dict[str, object]] = []
        if self._scene_project.regions:
            for region in self._scene_project.regions.values():
                if region.material_id is None or int(region.material_id) <= 0:
                    continue
                points = self._normalize_polygon_points(self._scene_region_points(region))
                if len(points) < 3:
                    continue
                faces.append({"points": points, "material_id": int(region.material_id)})
        else:
            for rule in self._material_geometry_region_rules.values():
                if not isinstance(rule, dict):
                    continue
                material_id = int(rule.get("material_id", 0))
                points_raw = rule.get("points", [])
                points = self._normalize_polygon_points(points_raw if isinstance(points_raw, list) else [])
                if material_id <= 0 or len(points) < 3:
                    continue
                faces.append({"points": points, "material_id": material_id})
        self._mesh_canvas.set_material_sketch_faces(faces)

    def _pick_sketch_face_region_at(self, x: float, y: float) -> dict[str, object] | None:
        if not self._sketch_face_pick_cache:
            self._rebuild_sketch_face_pick_cache()
        candidates = []
        px = float(x)
        py = float(y)
        for region in self._sketch_face_pick_cache:
            bbox = region.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            min_x, min_y, max_x, max_y = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            if px < min_x or px > max_x or py < min_y or py > max_y:
                continue
            points = region.get("points", [])
            if not isinstance(points, list):
                continue
            polygon = [(float(item[0]), float(item[1])) for item in points]
            if self._point_in_polygon((px, py), polygon):
                candidates.append(region)
        if not candidates:
            return None
        candidates.sort(key=lambda item: float(item.get("area", 0.0)))
        return candidates[0]

    @staticmethod
    def _distance_point_to_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float]:
        px, py = float(point[0]), float(point[1])
        x0, y0 = float(start[0]), float(start[1])
        x1, y1 = float(end[0]), float(end[1])
        dx = x1 - x0
        dy = y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 1e-12:
            return hypot(px - x0, py - y0), 0.0
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        return hypot(px - proj_x, py - proj_y), t

    def _pick_geometry_vertex_id_at(self, x: float, y: float, *, tolerance: float) -> str | None:
        self._ensure_geometry_pick_cache()
        best_id: str | None = None
        best_dist = float("inf")
        px = float(x)
        py = float(y)
        tol = float(tolerance)
        for point_id, point_x, point_y in self._geometry_pick_point_cache:
            if abs(px - point_x) > tol or abs(py - point_y) > tol:
                continue
            dist = hypot(px - point_x, py - point_y)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_id = point_id
        return best_id

    def _pick_geometry_edge_id_at(self, x: float, y: float, *, tolerance: float) -> tuple[str | None, float]:
        self._ensure_geometry_pick_cache()
        best_id: str | None = None
        best_param = 0.0
        best_dist = float("inf")
        px = float(x)
        py = float(y)
        tol = float(tolerance)
        for edge_id, x0, y0, x1, y1, min_x, min_y, max_x, max_y in self._geometry_pick_edge_cache:
            if px < (min_x - tol) or px > (max_x + tol) or py < (min_y - tol) or py > (max_y + tol):
                continue
            dist, param = self._distance_point_to_segment(
                (px, py),
                (x0, y0),
                (x1, y1),
            )
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_id = edge_id
                best_param = param
        return best_id, best_param

    def _create_geometry_edge_point_at(self, x: float, y: float, *, tolerance: float | None = None) -> str | None:
        tol = float(tolerance) if tolerance is not None else max(float(self._active_sketch_grid_step) * 0.20, 1e-4)
        edge_id, param = self._pick_geometry_edge_id_at(float(x), float(y), tolerance=tol)
        if edge_id is None:
            return None
        edge = self._scene_project.geometry_edges.get(edge_id)
        if edge is None:
            return None
        start = self._scene_project.geometry_points.get(edge.start_point_id)
        end = self._scene_project.geometry_points.get(edge.end_point_id)
        if start is None or end is None:
            return None
        t = max(0.0, min(1.0, float(param)))
        if t <= 1e-5:
            return edge.start_point_id
        if t >= 1.0 - 1e-5:
            return edge.end_point_id
        for point_id, point in self._scene_project.geometry_points.items():
            if point.owner_edge_id != edge_id:
                continue
            if point.role != "edge_point":
                continue
            if point.param is None:
                continue
            if abs(float(point.param) - t) <= 1e-4:
                return point_id
        px, py = self._interpolate_segment((float(start.x), float(start.y)), (float(end.x), float(end.y)), t)
        point_id = self._next_manual_geometry_point_id()
        self._scene_project.geometry_points[point_id] = GeometryPointDef(
            id=point_id,
            name=self._ui("边上点", "Edge Point"),
            x=float(px),
            y=float(py),
            owner_loop_id=edge.loop_id,
            owner_edge_id=edge_id,
            param=t,
            source="manual",
            role="edge_point",
        )
        self._rebuild_geometry_pick_cache()
        return point_id

    def _resolve_or_create_geometry_point_at(self, x: float, y: float) -> str | None:
        tolerance = max(float(self._active_sketch_grid_step) * 0.20, 1e-4)
        vertex_id = self._pick_geometry_vertex_id_at(float(x), float(y), tolerance=tolerance)
        if vertex_id is not None:
            return vertex_id
        edge_point_id = self._create_geometry_edge_point_at(float(x), float(y), tolerance=tolerance)
        if edge_point_id is not None:
            return edge_point_id
        region = self._pick_sketch_face_region_at(float(x), float(y))
        if region is None:
            return None
        point_id = self._next_manual_geometry_point_id()
        self._scene_project.geometry_points[point_id] = GeometryPointDef(
            id=point_id,
            name=self._ui("内部点", "Interior Point"),
            x=float(x),
            y=float(y),
            owner_loop_id=None,
            owner_edge_id=None,
            param=None,
            source="manual",
            role="interior_point",
        )
        self._rebuild_geometry_pick_cache()
        return point_id

    def _update_geometry_region_rules(self, material_id: int, selected_keys: set[str]) -> int:
        if not selected_keys:
            return 0
        updated = 0
        regions_by_key = {
            str(item["key"]): item
            for item in self._collect_sketch_face_regions()
        }
        for key in sorted(selected_keys):
            region = regions_by_key.get(key)
            if region is None:
                continue
            points = region.get("points", [])
            if not isinstance(points, list):
                continue
            rule = {
                "material_id": int(material_id),
                "points": [(float(x), float(y)) for x, y in points],
                "label": str(region.get("label", key)),
            }
            if key in self._material_geometry_region_rules:
                self._material_geometry_region_rules.pop(key)
            self._material_geometry_region_rules[key] = rule
            scene_region = self._scene_project.regions.get(self._scene_region_id_from_key(key))
            if scene_region is not None:
                scene_region.material_id = int(material_id)
            updated += 1
        self._sync_legacy_material_rules_from_scene()
        self._sync_component_sketch_faces_to_canvas()
        return updated

    def _apply_geometry_material_rules_to_mesh(self) -> int:
        if self._model is None or not self._model.mesh.elements:
            return 0
        if not self._material_geometry_region_rules and not self._scene_project.regions:
            return 0

        node_by_id = {node.id: node for node in self._model.mesh.nodes}
        rules: list[tuple[int, list[tuple[float, float]]]] = []
        if self._scene_project.regions:
            for region in self._scene_project.regions.values():
                if region.material_id is None or int(region.material_id) <= 0:
                    continue
                points = self._normalize_polygon_points(self._scene_region_points(region))
                if len(points) < 3:
                    continue
                rules.append((int(region.material_id), points))
        else:
            for key in self._material_geometry_region_rules:
                rule = self._material_geometry_region_rules.get(key)
                if not isinstance(rule, dict):
                    continue
                material_id = int(rule.get("material_id", 0))
                points_raw = rule.get("points", [])
                points = self._normalize_polygon_points(points_raw if isinstance(points_raw, list) else [])
                if material_id <= 0 or len(points) < 3:
                    continue
                rules.append((material_id, points))
        if not rules:
            return 0

        changed = 0
        for element in self._model.mesh.elements:
            if len(element.connectivity) != 3:
                continue
            nodes = [node_by_id.get(int(node_id)) for node_id in element.connectivity]
            if any(node is None for node in nodes):
                continue
            cx = (float(nodes[0].x) + float(nodes[1].x) + float(nodes[2].x)) / 3.0
            cy = (float(nodes[0].y) + float(nodes[1].y) + float(nodes[2].y)) / 3.0
            target_material_id: int | None = None
            for material_id, polygon in reversed(rules):
                if self._point_in_polygon((cx, cy), polygon):
                    target_material_id = int(material_id)
                    break
            if target_material_id is None:
                continue
            self._ensure_material_for_new_elements(target_material_id)
            if int(element.material_id) != target_material_id:
                element.material_id = target_material_id
                changed += 1
        self._rebuild_scene_mesh_state()
        return changed

    def _find_material_by_id(self, material_id: int) -> Material | None:
        if self._model is None:
            return None
        for material in self._model.materials:
            if int(material.id) == int(material_id):
                return material
        return None

    def _material_usage_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        if self._model is None:
            return counts
        for element in self._model.mesh.elements:
            mid = int(element.material_id)
            counts[mid] = counts.get(mid, 0) + 1
        return counts

    def _material_rule_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        if self._scene_project.regions:
            for region in self._scene_project.regions.values():
                points = self._normalize_polygon_points(self._scene_region_points(region))
                if len(points) < 3:
                    continue
                material_id = int(region.material_id or 0)
                if material_id <= 0:
                    continue
                counts[material_id] = counts.get(material_id, 0) + 1
        else:
            for rule in self._material_geometry_region_rules.values():
                if not isinstance(rule, dict):
                    continue
                material_id = int(rule.get("material_id", 0))
                if material_id <= 0:
                    continue
                counts[material_id] = counts.get(material_id, 0) + 1
        return counts

    def _material_display_name(self, material_id: int) -> str:
        return self._material_id_to_name.get(int(material_id), f"MAT-{int(material_id)}")

    def _material_mode_text(self, material: Material) -> str:
        return self._ui("平面应力", "Plane stress") if material.plane_stress else self._ui("平面应变", "Plane strain")

    def _material_combo_text(self, material: Material) -> str:
        return f"{self._material_display_name(material.id)} [ID={material.id}]"

    def _material_template_definitions(self) -> list[dict[str, object]]:
        return [
            {
                "key": "custom",
                "label": self._ui("自定义", "Custom"),
                "young_modulus": 2.0e7,
                "poisson_ratio": 0.30,
                "plane_stress": True,
                "hint": self._ui("保持当前输入，不自动覆盖。", "Keep current values without overwriting them."),
            },
            {
                "key": "soft_soil",
                "label": self._ui("软土", "Soft soil"),
                "young_modulus": 1.2e7,
                "poisson_ratio": 0.35,
                "plane_stress": False,
                "hint": self._ui("适合软土/回填土的起始参数。", "Starter values for soft soil and backfill."),
            },
            {
                "key": "clay",
                "label": self._ui("黏土", "Clay"),
                "young_modulus": 2.5e7,
                "poisson_ratio": 0.32,
                "plane_stress": False,
                "hint": self._ui("适合一般黏性土的二维平面应变起始参数。", "Starter values for typical clay in 2D plane-strain analysis."),
            },
            {
                "key": "sand",
                "label": self._ui("砂土", "Sand"),
                "young_modulus": 4.5e7,
                "poisson_ratio": 0.30,
                "plane_stress": False,
                "hint": self._ui("适合中密砂土的初始估算参数。", "Initial estimate for medium-dense sand."),
            },
            {
                "key": "dense_soil",
                "label": self._ui("密实土", "Dense soil"),
                "young_modulus": 6.0e7,
                "poisson_ratio": 0.28,
                "plane_stress": False,
                "hint": self._ui("适合中密到密实土层的起始参数。", "Starter values for medium-dense to dense soils."),
            },
            {
                "key": "rock_mass",
                "label": self._ui("岩体", "Rock mass"),
                "young_modulus": 2.5e10,
                "poisson_ratio": 0.24,
                "plane_stress": False,
                "hint": self._ui("适合岩体或强风化以下岩层的起始参数。", "Starter values for rock mass analyses."),
            },
            {
                "key": "weathered_rock",
                "label": self._ui("强风化岩", "Weathered rock"),
                "young_modulus": 8.0e8,
                "poisson_ratio": 0.26,
                "plane_stress": False,
                "hint": self._ui("适合强风化岩或破碎岩体的初始参数。", "Starter values for weathered or fractured rock."),
            },
            {
                "key": "intact_rock",
                "label": self._ui("完整岩石", "Intact rock"),
                "young_modulus": 4.0e10,
                "poisson_ratio": 0.22,
                "plane_stress": False,
                "hint": self._ui("适合较完整岩石材料的起始参数。", "Starter values for relatively intact rock."),
            },
            {
                "key": "concrete",
                "label": self._ui("混凝土", "Concrete"),
                "young_modulus": 3.0e10,
                "poisson_ratio": 0.20,
                "plane_stress": True,
                "hint": self._ui("常见混凝土二维分析起始参数。", "Typical starting values for 2D concrete analysis."),
            },
            {
                "key": "steel",
                "label": self._ui("钢材", "Steel"),
                "young_modulus": 2.06e11,
                "poisson_ratio": 0.30,
                "plane_stress": True,
                "hint": self._ui("常见钢材二维分析起始参数。", "Typical starting values for 2D steel analysis."),
            },
        ]

    def _material_template_combo_widgets(self) -> list[QComboBox]:
        combos: list[QComboBox] = []
        for name in (
            "_material_template_combo",
            "_material_toolbar_template_combo",
            "_material_action_template_combo",
        ):
            combo = getattr(self, name, None)
            if isinstance(combo, QComboBox):
                combos.append(combo)
        return combos

    def _material_template_by_key(self, key: str) -> dict[str, object] | None:
        for item in self._material_template_definitions():
            if str(item.get("key", "")) == str(key):
                return item
        return None

    def _sync_material_template_combo_selection(self, key: str, source: QComboBox | None = None) -> None:
        for combo in self._material_template_combo_widgets():
            if combo is source:
                continue
            combo.blockSignals(True)
            index = combo.findData(str(key))
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def _selected_material_template(self) -> dict[str, object] | None:
        key = ""
        for combo in self._material_template_combo_widgets():
            combo_key = str(combo.currentData() or "")
            if combo_key:
                key = combo_key
                break
        return self._material_template_by_key(key)

    def _on_material_template_changed(self, _: int) -> None:
        source = self.sender()
        key = ""
        if isinstance(source, QComboBox):
            key = str(source.currentData() or "")
            if not self._is_syncing_material_widgets:
                self._sync_material_template_combo_selection(key, source)
        template = self._material_template_by_key(key) if key else self._selected_material_template()
        if template is None:
            return
        hint = str(template.get("hint", "")).strip()
        self._material_unit_hint.setText(
            self._ui(
                f"单位建议：E 使用 Pa，nu 无量纲。{hint}",
                f"Units: use Pa for E and dimensionless nu. {hint}",
            )
        )

    def _apply_selected_material_template(self) -> None:
        template = self._selected_material_template()
        if template is None or str(template.get("key", "")) == "custom":
            return
        self._material_e_spin.setValue(float(template.get("young_modulus", self._material_e_spin.value())))
        self._material_nu_spin.setValue(float(template.get("poisson_ratio", self._material_nu_spin.value())))
        self._material_plane_stress.setChecked(bool(template.get("plane_stress", self._material_plane_stress.isChecked())))

    def _material_region_entries(self, material_id: int) -> list[RegionDef]:
        regions: list[RegionDef] = []
        for region in self._scene_project.regions.values():
            if region.material_id is None:
                continue
            if int(region.material_id) == int(material_id):
                regions.append(region)
        regions.sort(key=lambda item: item.name)
        return regions

    def _material_related_set_entries(self, material_id: int) -> list[GeometrySetDef]:
        regions = self._material_region_entries(material_id)
        if not regions:
            return []
        region_ids = {region.id for region in regions}
        component_ids = {region.component_id for region in regions}
        related_sets: list[GeometrySetDef] = []
        for geometry_set in self._scene_project.geometry_sets.values():
            related = False
            if geometry_set.entity_type == "region":
                related = any(region_id in region_ids for region_id in geometry_set.entity_ids)
            elif geometry_set.entity_type == "component":
                related = any(component_id in component_ids for component_id in geometry_set.entity_ids)
            if related:
                related_sets.append(geometry_set)
        related_sets.sort(key=lambda item: (item.entity_type, item.name))
        return related_sets

    def _material_usage_details_text(self, material_id: int) -> str:
        material = self._find_material_by_id(material_id)
        if material is None:
            return self._ui("未选择材料。", "No material selected.")
        element_ids = []
        if self._model is not None:
            element_ids = [
                int(element.id)
                for element in self._model.mesh.elements
                if int(element.material_id) == int(material_id)
            ]
        regions = self._material_region_entries(material_id)
        region_names = [f"{region.name} [{region.id}]" for region in regions]
        component_names = sorted(
            {
                self._scene_project.components.get(region.component_id).name
                if self._scene_project.components.get(region.component_id) is not None
                else region.component_id
                for region in regions
            }
        )
        related_sets = self._material_related_set_entries(material_id)
        set_names = [f"{item.name} [{item.entity_type}]" for item in related_sets]
        element_preview = ", ".join(str(item) for item in element_ids[:12])
        if len(element_ids) > 12:
            element_preview += ", ..."
        region_preview = ", ".join(region_names[:8])
        if len(region_names) > 8:
            region_preview += ", ..."
        component_preview = ", ".join(component_names[:6])
        if len(component_names) > 6:
            component_preview += ", ..."
        set_preview = ", ".join(set_names[:6])
        if len(set_names) > 6:
            set_preview += ", ..."
        return self._ui(
            "\n".join(
                [
                    f"材料: {self._material_display_name(material.id)} [ID={material.id}]",
                    f"参数: E={float(material.young_modulus):.4e} Pa, nu={float(material.poisson_ratio):.4f}, {self._material_mode_text(material)}",
                    f"单元覆盖: {len(element_ids)} 个" + (f" -> {element_preview}" if element_preview else ""),
                    f"草图区域绑定: {len(regions)} 个" + (f" -> {region_preview}" if region_preview else ""),
                    f"所属组件: {len(component_names)} 个" + (f" -> {component_preview}" if component_preview else ""),
                    f"关联集合: {len(related_sets)} 个" + (f" -> {set_preview}" if set_preview else ""),
                ]
            ),
            "\n".join(
                [
                    f"Material: {self._material_display_name(material.id)} [ID={material.id}]",
                    f"Params: E={float(material.young_modulus):.4e} Pa, nu={float(material.poisson_ratio):.4f}, {self._material_mode_text(material)}",
                    f"Element coverage: {len(element_ids)}" + (f" -> {element_preview}" if element_preview else ""),
                    f"Sketch-region bindings: {len(regions)}" + (f" -> {region_preview}" if region_preview else ""),
                    f"Components: {len(component_names)}" + (f" -> {component_preview}" if component_preview else ""),
                    f"Related sets: {len(related_sets)}" + (f" -> {set_preview}" if set_preview else ""),
                ]
            ),
        )

    def _refresh_material_usage_details(self, material_id: int | None = None) -> None:
        target_id = int(material_id) if material_id is not None else int(self._material_assign_id_spin.value())
        if hasattr(self, "_material_usage_details"):
            self._material_usage_details.setPlainText(self._material_usage_details_text(target_id))

    def _highlight_selected_material_usage(self, material_id: int) -> None:
        if self._pending_pick_context in {"material_face", "material_face_sketch", "component_face", "component_face_sketch"}:
            return
        if self._model is not None and self._model.mesh.elements:
            highlighted_element_ids = {
                int(element.id)
                for element in self._model.mesh.elements
                if int(element.material_id) == int(material_id)
            }
            self._selected_element_ids = set(highlighted_element_ids)
            self._mesh_canvas.set_highlighted_elements(highlighted_element_ids)
        sketch_faces = []
        for region in self._material_region_entries(material_id):
            points = self._normalize_polygon_points(self._scene_region_points(region))
            if len(points) >= 3:
                sketch_faces.append(points)
        self._mesh_canvas.set_highlighted_sketch_faces(sketch_faces)

    def _material_manager_selected_id(self) -> int | None:
        if not hasattr(self, "_material_manager_table"):
            return None
        row = self._material_manager_table.currentRow()
        if row < 0:
            return None
        item = self._material_manager_table.item(row, 0)
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        if data is None:
            return None
        return int(data)

    def _update_material_summary_labels(
        self,
        selected_material_id: int | None = None,
        *,
        usage_counts: dict[int, int] | None = None,
        rule_counts: dict[int, int] | None = None,
    ) -> None:
        usage = usage_counts if usage_counts is not None else self._material_usage_counts()
        rules = rule_counts if rule_counts is not None else self._material_rule_counts()
        material_count = len(self._model.materials) if self._model is not None else 0
        total_elements = sum(usage.values())
        total_rules = sum(rules.values())

        if material_count <= 0:
            manager_text = self._ui(
                "暂无材料。创建后会自动出现在此处，并同步到分配材料下拉框与顶部工具条。",
                "No materials yet. New materials will appear here, in the assignment dropdown, and in the top material strip.",
            )
            toolbar_text = self._ui("请先创建材料", "Create a material first")
        else:
            current_id = int(selected_material_id) if selected_material_id is not None else int(self._material_assign_id_spin.value())
            current_name = self._material_display_name(current_id)
            current_usage = usage.get(current_id, 0)
            current_rules = rules.get(current_id, 0)
            manager_text = self._ui(
                f"共 {material_count} 个材料，累计覆盖 {total_elements} 个单元，预绑定 {total_rules} 个草图区域。当前选中 {current_name} [ID={current_id}]，已分配 {current_usage} 个单元，预绑定 {current_rules} 个区域。",
                f"{material_count} materials total, covering {total_elements} elements with {total_rules} sketch-region bindings. Current: {current_name} [ID={current_id}], assigned to {current_usage} elements with {current_rules} pending region bindings.",
            )
            toolbar_text = self._ui(
                f"{current_name}：{current_usage} 单元，{current_rules} 区域",
                f"{current_name}: {current_usage} elems, {current_rules} regions",
            )

        if hasattr(self, "_material_manager_summary"):
            self._material_manager_summary.setText(manager_text)
        if hasattr(self, "_material_toolbar_summary"):
            self._material_toolbar_summary.setText(toolbar_text)

    def _sync_material_controls(self) -> None:
        materials = sorted(self._model.materials, key=lambda item: int(item.id)) if self._model is not None else []
        material_ids = [int(item.id) for item in materials]
        selected_material_id = int(self._material_assign_id_spin.value()) if hasattr(self, "_material_assign_id_spin") else 1
        if material_ids and selected_material_id not in material_ids:
            selected_material_id = material_ids[-1]
        if not material_ids:
            selected_material_id = 1

        usage_counts = self._material_usage_counts()
        rule_counts = self._material_rule_counts()
        has_materials = bool(material_ids)

        self._is_syncing_material_widgets = True
        try:
            if hasattr(self, "_material_assign_id_spin"):
                self._material_assign_id_spin.blockSignals(True)
                self._material_assign_id_spin.setValue(selected_material_id)
                self._material_assign_id_spin.blockSignals(False)

            for combo_name in ("_material_assign_combo", "_material_toolbar_combo", "_material_reassign_target_combo"):
                combo = getattr(self, combo_name, None)
                if combo is None:
                    continue
                combo.blockSignals(True)
                combo.clear()
                if has_materials:
                    for material in materials:
                        combo.addItem(self._material_combo_text(material), int(material.id))
                    preferred_id = selected_material_id
                    if combo_name == "_material_reassign_target_combo" and len(material_ids) >= 2 and preferred_id == selected_material_id:
                        preferred_id = next((mid for mid in material_ids if mid != selected_material_id), selected_material_id)
                    current_index = combo.findData(preferred_id)
                    if current_index >= 0:
                        combo.setCurrentIndex(current_index)
                else:
                    combo.addItem(self._tr("rock.material.none", "No materials yet"), 0)
                    combo.setCurrentIndex(0)
                combo.setEnabled(has_materials)
                combo.blockSignals(False)

            template_combos = self._material_template_combo_widgets()
            current_template_key = "custom"
            for combo in template_combos:
                combo_key = str(combo.currentData() or "")
                if combo_key:
                    current_template_key = combo_key
                    break
            for combo in template_combos:
                combo.blockSignals(True)
                combo.clear()
                for template in self._material_template_definitions():
                    combo.addItem(str(template.get("label", "")), str(template.get("key", "")))
                template_index = combo.findData(current_template_key)
                if template_index < 0:
                    template_index = combo.findData("custom")
                combo.setCurrentIndex(max(template_index, 0))
                combo.setEnabled(True)
                combo.blockSignals(False)

            if hasattr(self, "_material_manager_table"):
                self._material_manager_table.blockSignals(True)
                self._material_manager_table.setRowCount(len(materials))
                selected_row = -1
                for row, material in enumerate(materials):
                    values = [
                        str(material.id),
                        self._material_display_name(material.id),
                        f"{float(material.young_modulus):.4e}",
                        f"{float(material.poisson_ratio):.4f}",
                        self._material_mode_text(material),
                        str(usage_counts.get(int(material.id), 0)),
                    ]
                    for col, text in enumerate(values):
                        item = QTableWidgetItem(text)
                        item.setData(Qt.UserRole, int(material.id))
                        self._material_manager_table.setItem(row, col, item)
                    if int(material.id) == selected_material_id:
                        selected_row = row
                if selected_row >= 0:
                    self._material_manager_table.selectRow(selected_row)
                else:
                    self._material_manager_table.clearSelection()
                self._material_manager_table.blockSignals(False)

            if has_materials:
                material = self._find_material_by_id(selected_material_id)
                if material is not None:
                    self._material_name_edit.setText(self._material_display_name(material.id))
                    self._material_e_spin.setValue(float(material.young_modulus))
                    self._material_nu_spin.setValue(float(material.poisson_ratio))
                    self._material_plane_stress.setChecked(bool(material.plane_stress))
            else:
                self._material_name_edit.setText("Soil-1")
                self._material_e_spin.setValue(2.0e7)
                self._material_nu_spin.setValue(0.30)
                self._material_plane_stress.setChecked(True)
            self._refresh_material_usage_details(selected_material_id if has_materials else None)
        finally:
            self._is_syncing_material_widgets = False

        if hasattr(self, "_btn_material_delete"):
            self._btn_material_delete.setEnabled(has_materials)
        if hasattr(self, "_btn_material_rename"):
            self._btn_material_rename.setEnabled(has_materials)
        if hasattr(self, "_btn_material_duplicate"):
            self._btn_material_duplicate.setEnabled(has_materials)
        if hasattr(self, "_btn_material_tool_delete"):
            self._btn_material_tool_delete.setEnabled(has_materials)
        if hasattr(self, "_btn_material_tool_rename"):
            self._btn_material_tool_rename.setEnabled(has_materials)
        if hasattr(self, "_btn_material_tool_duplicate"):
            self._btn_material_tool_duplicate.setEnabled(has_materials)
        if hasattr(self, "_btn_material_tool_assign"):
            self._btn_material_tool_assign.setEnabled(has_materials)
        if hasattr(self, "_btn_material_action_rename"):
            self._btn_material_action_rename.setEnabled(has_materials)
        if hasattr(self, "_btn_material_action_duplicate"):
            self._btn_material_action_duplicate.setEnabled(has_materials)
        if hasattr(self, "_btn_material_action_delete"):
            self._btn_material_action_delete.setEnabled(has_materials)
        if hasattr(self, "_btn_material_action_assign"):
            self._btn_material_action_assign.setEnabled(has_materials)
        if hasattr(self, "_btn_material_reassign_all"):
            self._btn_material_reassign_all.setEnabled(has_materials and len(material_ids) >= 2)
        self._update_material_summary_labels(
            selected_material_id if has_materials else None,
            usage_counts=usage_counts,
            rule_counts=rule_counts,
        )

    def _select_material_by_id(self, material_id: int, *, load_form: bool = True) -> None:
        if self._is_syncing_material_widgets:
            return
        material = self._find_material_by_id(material_id)
        if material is None:
            return

        self._is_syncing_material_widgets = True
        try:
            self._material_assign_id_spin.blockSignals(True)
            self._material_assign_id_spin.setValue(int(material.id))
            self._material_assign_id_spin.blockSignals(False)

            for combo_name in ("_material_assign_combo", "_material_toolbar_combo"):
                combo = getattr(self, combo_name, None)
                if combo is None:
                    continue
                combo.blockSignals(True)
                index = combo.findData(int(material.id))
                if index >= 0:
                    combo.setCurrentIndex(index)
                combo.blockSignals(False)

            if hasattr(self, "_material_reassign_target_combo"):
                self._material_reassign_target_combo.blockSignals(True)
                target_id = next(
                    (
                        int(self._material_reassign_target_combo.itemData(index))
                        for index in range(self._material_reassign_target_combo.count())
                        if int(self._material_reassign_target_combo.itemData(index) or 0) != int(material.id)
                    ),
                    int(material.id),
                )
                target_index = self._material_reassign_target_combo.findData(target_id)
                if target_index >= 0:
                    self._material_reassign_target_combo.setCurrentIndex(target_index)
                self._material_reassign_target_combo.blockSignals(False)

            if hasattr(self, "_material_manager_table"):
                self._material_manager_table.blockSignals(True)
                target_row = -1
                for row in range(self._material_manager_table.rowCount()):
                    item = self._material_manager_table.item(row, 0)
                    if item is None:
                        continue
                    if int(item.data(Qt.UserRole)) == int(material.id):
                        target_row = row
                        break
                if target_row >= 0:
                    self._material_manager_table.selectRow(target_row)
                self._material_manager_table.blockSignals(False)

            if load_form:
                self._material_name_edit.setText(self._material_display_name(material.id))
                self._material_e_spin.setValue(float(material.young_modulus))
                self._material_nu_spin.setValue(float(material.poisson_ratio))
                self._material_plane_stress.setChecked(bool(material.plane_stress))
        finally:
            self._is_syncing_material_widgets = False

        self._update_material_summary_labels(int(material.id))
        self._refresh_material_usage_details(int(material.id))
        self._highlight_selected_material_usage(int(material.id))

    def _on_material_assign_combo_changed(self, _: int) -> None:
        if self._is_syncing_material_widgets:
            return
        material_id = int(self._material_assign_combo.currentData() or 0)
        if material_id > 0:
            self._select_material_by_id(material_id)

    def _on_material_toolbar_combo_changed(self, _: int) -> None:
        if self._is_syncing_material_widgets:
            return
        material_id = int(self._material_toolbar_combo.currentData() or 0)
        if material_id > 0:
            self._select_material_by_id(material_id)

    def _on_material_manager_selection_changed(self) -> None:
        if self._is_syncing_material_widgets:
            return
        material_id = self._material_manager_selected_id()
        if material_id is not None and material_id > 0:
            self._select_material_by_id(material_id)

    def _rename_selected_material(self) -> None:
        material_id = int(self._material_assign_id_spin.value())
        material = self._find_material_by_id(material_id)
        if material is None:
            return
        current_name = self._material_display_name(material_id)
        new_name, ok = QInputDialog.getText(
            self,
            self._ui("重命名材料", "Rename Material"),
            self._ui("新名称", "New Name"),
            text=current_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == current_name:
            return
        existing_id = self._material_name_to_id.get(new_name)
        if existing_id is not None and int(existing_id) != material_id:
            self._show_error(
                self._ui("重命名材料失败", "Rename material failed"),
                self._ui("该材料名称已存在，请换一个名称。", "That material name already exists. Please choose another one."),
            )
            return
        self._material_id_to_name[material_id] = new_name
        self._material_name_to_id = {
            name: mid
            for name, mid in self._material_name_to_id.items()
            if int(mid) != material_id
        }
        self._material_name_to_id[new_name] = material_id
        self._sync_material_controls()
        self._select_material_by_id(material_id)
        self._log(
            self._ui(
                f"材料已重命名: {current_name} -> {new_name} [ID={material_id}]。",
                f"Material renamed: {current_name} -> {new_name} [ID={material_id}].",
            )
        )

    def _duplicate_selected_material(self) -> None:
        if self._model is None:
            return
        material_id = int(self._material_assign_id_spin.value())
        material = self._find_material_by_id(material_id)
        if material is None:
            return
        base_name = self._material_display_name(material_id)
        suggested_name = f"{base_name}-Copy"
        new_name, ok = QInputDialog.getText(
            self,
            self._ui("复制材料", "Duplicate Material"),
            self._ui("新材料名称", "New material name"),
            text=suggested_name,
        )
        if not ok:
            return
        new_name = new_name.strip() or suggested_name
        if new_name in self._material_name_to_id:
            self._show_error(
                self._ui("复制材料失败", "Duplicate material failed"),
                self._ui("该材料名称已存在，请换一个名称。", "That material name already exists. Please choose another one."),
            )
            return
        new_id = max((int(item.id) for item in self._model.materials), default=0) + 1
        clone = Material(
            id=new_id,
            young_modulus=float(material.young_modulus),
            poisson_ratio=float(material.poisson_ratio),
            plane_stress=bool(material.plane_stress),
        )
        self._material_id_to_name[new_id] = new_name
        self._material_name_to_id[new_name] = new_id
        self._model.materials.append(clone)
        self._model.materials.sort(key=lambda item: int(item.id))
        self._set_model(self._model)
        self._select_material_by_id(new_id)
        self._log(
            self._ui(
                f"已复制材料 {base_name}，新材料为 {new_name} [ID={new_id}]。",
                f"Duplicated material {base_name}; new material is {new_name} [ID={new_id}].",
            )
        )

    def _batch_reassign_selected_material(self) -> None:
        if self._model is None:
            return
        source_id = int(self._material_assign_id_spin.value())
        target_id = int(self._material_reassign_target_combo.currentData() or 0)
        if source_id <= 0 or target_id <= 0 or source_id == target_id:
            return
        source_name = self._material_display_name(source_id)
        target_name = self._material_display_name(target_id)
        changed_elements = 0
        for element in self._model.mesh.elements:
            if int(element.material_id) == source_id:
                element.material_id = target_id
                changed_elements += 1
        changed_regions = 0
        for region in self._scene_project.regions.values():
            if region.material_id is None:
                continue
            if int(region.material_id) == source_id:
                region.material_id = target_id
                changed_regions += 1
        self._sync_legacy_material_rules_from_scene()
        self._set_model(self._model)
        self._select_material_by_id(target_id)
        if changed_elements > 0:
            self._mark_results_stale("Material bulk reassignment updated")
        self._log(
            self._ui(
                f"已将材料 {source_name} 批量重分配到 {target_name}：单元 {changed_elements} 个，草图区域 {changed_regions} 个。",
                f"Bulk reassigned material {source_name} to {target_name}: {changed_elements} elements and {changed_regions} sketch regions.",
            )
        )

    def _delete_selected_material(self) -> None:
        if self._model is None:
            return
        material_id = int(self._material_assign_id_spin.value())
        material = self._find_material_by_id(material_id)
        if material is None:
            return

        usage_counts = self._material_usage_counts()
        used_count = usage_counts.get(material_id, 0)
        if used_count > 0:
            self._show_error(
                self._ui("删除材料失败", "Delete material failed"),
                self._ui(
                    f"材料 {self._material_display_name(material_id)} 仍被 {used_count} 个单元使用，请先重新分配这些单元。",
                    f"Material {self._material_display_name(material_id)} is still used by {used_count} elements. Reassign those elements before deleting it.",
                ),
            )
            return

        removed_rule_count = 0
        for key in list(self._material_geometry_region_rules):
            rule = self._material_geometry_region_rules.get(key)
            if not isinstance(rule, dict):
                continue
            if int(rule.get("material_id", 0)) == material_id:
                self._material_geometry_region_rules.pop(key)
                removed_rule_count += 1
        for region in self._scene_project.regions.values():
            if region.material_id is not None and int(region.material_id) == material_id:
                region.material_id = None

        name = self._material_display_name(material_id)
        self._model.materials = [item for item in self._model.materials if int(item.id) != material_id]
        self._material_id_to_name.pop(material_id, None)
        self._material_name_to_id = {
            item_name: mid
            for item_name, mid in self._material_name_to_id.items()
            if int(mid) != material_id
        }
        self._set_model(self._model)
        self._log(
            self._ui(
                f"已删除材料 {name} [ID={material_id}]，同时清理 {removed_rule_count} 个草图区域绑定。",
                f"Deleted material {name} [ID={material_id}] and removed {removed_rule_count} sketch-region bindings.",
            )
        )

    def _create_isotropic_material(self) -> None:
        if self._model is None:
            self._create_empty_model()
        if self._model is None:
            return
        name = self._material_name_edit.text().strip()
        if not name:
            fallback_id = max((m.id for m in self._model.materials), default=0) + 1
            name = f"MAT-{fallback_id}"
            self._material_name_edit.setText(name)

        material_id = self._material_name_to_id.get(name)
        if material_id is None:
            material_id = max((item.id for item in self._model.materials), default=0) + 1
        self._material_name_to_id[name] = material_id
        self._material_id_to_name[material_id] = name

        material = Material(
            id=int(material_id),
            young_modulus=float(self._material_e_spin.value()),
            poisson_ratio=float(self._material_nu_spin.value()),
            plane_stress=bool(self._material_plane_stress.isChecked()),
        )
        self._model.materials = [item for item in self._model.materials if item.id != material.id]
        self._model.materials.append(material)
        self._model.materials.sort(key=lambda item: item.id)
        self._set_model(self._model)
        self._select_material_by_id(material.id)
        self._log(
            self._ui(
                f"材料已创建/更新: {name} (ID={material.id}, E={material.young_modulus:.4e}, nu={material.poisson_ratio:.4f})。",
                f"Material created/updated: {name} (ID={material.id}, E={material.young_modulus:.4e}, nu={material.poisson_ratio:.4f}).",
            )
        )

    def _assign_material_to_region(self) -> None:
        if self._model is None:
            return
        material_id = int(self._material_assign_id_spin.value())
        self._ensure_material_for_new_elements(material_id)
        raw = self._material_assign_elements.text().strip()
        selected_ids: set[int] = set()
        selected_sketch_keys: set[str] = set(self._selected_face_region_sketch_keys)
        if self._selected_face_region_element_ids:
            selected_ids = set(self._selected_face_region_element_ids)
        if raw:
            for token in raw.replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    selected_ids.add(int(token))
                except ValueError:
                    pass
        if not selected_ids:
            selected_ids = set(self._selected_element_ids)
        if selected_ids:
            changed = 0
            for element in self._model.mesh.elements:
                if element.id in selected_ids:
                    element.material_id = material_id
                    changed += 1
            if changed == 0:
                self._show_error(self._ui("材料分配失败", "Assign material failed"), self._ui("未找到匹配的单元。", "No matching elements found."))
                return
            self._set_model(self._model)
            self._mark_results_stale("Material assignment updated")
            self._selected_face_region_element_ids.clear()
            self._mesh_canvas.set_highlighted_elements(set())
            self._material_face_hint.setText(
                self._ui(f"已分配 {changed} 个单元", f"Assigned {changed} elements")
            )
            self._log(
                self._ui(
                    f"材料 ID={material_id} 已分配到 {changed} 个单元。",
                    f"Material {material_id} assigned to {changed} elements.",
                )
            )
            return

        if selected_sketch_keys:
            rule_count = self._update_geometry_region_rules(material_id, selected_sketch_keys)
            if rule_count <= 0:
                self._show_error(
                    self._ui("材料分配失败", "Assign material failed"),
                    self._ui("草图区域无效或已失效，请重新选择。", "Selected sketch region is invalid or outdated. Please pick again."),
                )
                return
            changed = self._apply_geometry_material_rules_to_mesh()
            if changed > 0:
                self._set_model(self._model)
            self._sync_material_sketch_faces_to_canvas()
            self._update_sketch_face_highlight()
            if changed <= 0:
                self._update_model_tree()
                self._update_status_labels()
            self._material_face_hint.setText(
                self._ui(
                    f"已绑定 {rule_count} 个草图区域（当前映射 {changed} 个单元）",
                    f"Bound {rule_count} sketch region(s) (mapped {changed} elements now)",
                )
            )
            self._log(
                self._ui(
                    f"材料 ID={material_id} 已绑定到 {rule_count} 个草图区域；当前网格映射单元 {changed} 个。",
                    f"Material {material_id} bound to {rule_count} sketch region(s); mapped {changed} mesh elements now.",
                )
            )
            return

        self._show_error(
            self._ui("材料分配失败", "Assign material failed"),
            self._ui("未选择区域。请先点击“选择面区域”并在画布中点选。", "No region selected. Click 'Pick Face Region' and select on canvas."),
        )

    def _apply_assembly_offset(self) -> None:
        if self._model is None:
            return
        dx = float(self._assembly_dx_spin.value())
        dy = float(self._assembly_dy_spin.value())
        if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
            return
        for node in self._model.mesh.nodes:
            node.x += dx
            node.y += dy
        self._set_model(self._model)
        self._mark_results_stale("Assembly transform applied")
        self._log(
            f"Assembly offset applied for {self._assembly_instance_name.text().strip() or 'PART-1'}: "
            f"dx={dx:.4f}, dy={dy:.4f}."
        )

    def _add_concentrated_load(self) -> None:
        if self._model is None:
            self._create_empty_model()
        if self._model is None:
            return
        if self._current_load_bc_target_mode() == "geometry":
            if self._current_load_bc_target_set_id() is None and self._selected_geometry_point_ids:
                preferred_set_id = self._preferred_geometry_set_id_for_pick("point")
                self._upsert_geometry_set_from_geometry_ids(
                    entity_type="point",
                    geometry_ids=self._selected_geometry_point_ids,
                    preferred_name=self._ui("几何点集合", "Geometry Point Set"),
                    preferred_set_id=preferred_set_id,
                )
            set_id = self._current_load_bc_target_set_id()
            target_set = self._scene_project.geometry_sets.get(set_id or "")
            if target_set is None:
                self._show_error(
                    self._ui("载荷失败", "Load failed"),
                    self._ui("集中力需要点集。请先点击“几何选点”并在画布中选择作用点。", "Concentrated load needs a point set. Click Pick Geometry Point and select a target first."),
                )
                return
            if target_set.entity_type != "point":
                self._show_error(
                    self._ui("载荷失败", "Load failed"),
                    self._ui("当前选择的不是点集。集中力只能施加到几何点，请点击“几何选点”。", "The selected target is not a point set. Use Pick Geometry Point for concentrated loads."),
                )
                return
            vx = float(self._load_point_vec_x.value())
            vy = float(self._load_point_vec_y.value())
            magnitude = float(self._load_point_magnitude_spin.value())
            if hypot(vx, vy) <= 1e-12:
                self._show_error(
                    self._ui("载荷失败", "Load failed"),
                    self._ui("方向向量不能为零，请输入有效方向。", "Direction vector cannot be zero."),
                )
                return
            load_id = self._next_load_definition_id()
            self._scene_project.load_definitions[load_id] = LoadDefinition(
                id=load_id,
                name=self._consume_pending_load_name("concentrated"),
                target_set_id=target_set.id,
                target_entity_type="point",
                load_type="concentrated",
                vector_x=vx,
                vector_y=vy,
                magnitude=magnitude,
                profile="uniform",
                step=self._consume_pending_load_step(),
            )
            unresolved = self._resolve_scene_load_bc_definitions_to_model() if self._model.mesh.nodes else []
            self._sync_load_bc_geometry_overlay()
            self._refresh_load_bc_managers()
            self._sync_editor_tables_from_model()
            self._update_model_tree()
            self._update_status_labels()
            self._mark_results_stale("Geometry concentrated load definition updated")
            if unresolved:
                self._load_bc_set_hint.setText(
                    self._ui(
                        "集中力已保存为几何载荷；当前网格尚未解析该点，生成保点网格后会自动展开为节点力。",
                        "Concentrated load saved on geometry; it will resolve to nodal force after geometry-preserving meshing.",
                    )
                )
            else:
                self._load_bc_set_hint.setText(
                    self._ui(
                        f"集中力已保存到几何点集 {target_set.name}，可先继续建模/划分网格。",
                        f"Concentrated load saved on geometry point set {target_set.name}.",
                    )
                )
            self._log(
                self._ui(
                    f"几何集中力已创建: 集合={target_set.name}, 方向=({vx:.4g},{vy:.4g}), 大小={magnitude:.6g}。",
                    f"Geometry concentrated load created: set={target_set.name}, dir=({vx:.4g},{vy:.4g}), mag={magnitude:.6g}.",
                )
            )
            return
        set_nodes = self._selected_load_bc_set_node_ids()
        if self._current_load_bc_target_mode() == "geometry" and not set_nodes:
            if self._current_load_bc_target_set_id() is None:
                self._show_error(
                    self._ui("载荷失败", "Load failed"),
                    self._ui("几何模式下请先创建/选择几何点集合。", "In geometry mode, please create/select a geometry point set first."),
                )
                return
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui("几何目标尚未解析到网格节点。请先生成网格并确保几何点被保留。", "Geometry target is unresolved to mesh nodes. Generate mesh and ensure point preservation first."),
            )
            return
        if len(set_nodes) > 1:
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui("集中力目标集合只能包含 1 个节点。", "Point load target set must contain exactly one node."),
            )
            return
        node_id = int(set_nodes[0]) if set_nodes else int(self._load_point_node_spin.value())
        self._load_point_node_spin.setValue(node_id)
        if self._get_node_by_id(node_id) is None:
            self._show_error(self._ui("载荷失败", "Load failed"), self._ui(f"节点 {node_id} 不存在。", f"Node {node_id} does not exist."))
            return
        vx = float(self._load_point_vec_x.value())
        vy = float(self._load_point_vec_y.value())
        magnitude = float(self._load_point_magnitude_spin.value())
        norm = hypot(vx, vy)
        if norm <= 1e-12:
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui("方向向量不能为零，请输入有效方向。", "Direction vector cannot be zero."),
            )
            return
        fx = magnitude * vx / norm
        fy = magnitude * vy / norm
        if abs(fx) > 1e-12:
            self._model.loads.append(Load(node_id=node_id, dof="fx", value=fx))
        if abs(fy) > 1e-12:
            self._model.loads.append(Load(node_id=node_id, dof="fy", value=fy))
        self._set_model(self._model)
        self._upsert_geometry_set_from_node_ids(
            entity_type="point",
            node_ids=[node_id],
            preferred_name=self._ui("点载荷集合", "Point Load Set"),
        )
        self._mark_results_stale("Concentrated load updated")
        self._log(
            self._ui(
                f"集中力已添加: 节点={node_id}, 方向=({vx:.4g}, {vy:.4g}), 大小={magnitude:.6g}, 分解=({fx:.6g}, {fy:.6g})。",
                f"Concentrated load added: node={node_id}, dir=({vx:.4g}, {vy:.4g}), mag={magnitude:.6g}, components=({fx:.6g}, {fy:.6g}).",
            )
        )

    def _add_distributed_load(self) -> None:
        if self._model is None:
            self._create_empty_model()
        if self._model is None:
            return
        load_type = str(self._load_dist_type_combo.currentData()) if hasattr(self, "_load_dist_type_combo") else "distributed"
        direction_mode = str(self._load_dist_direction_mode_combo.currentData()) if hasattr(self, "_load_dist_direction_mode_combo") else "vector"
        axis = str(self._load_dist_axis_combo.currentData())
        coord = float(self._load_dist_coord_spin.value())
        tol = float(self._load_dist_tol_spin.value())
        vx = float(self._load_dist_vec_x.value())
        vy = float(self._load_dist_vec_y.value())
        profile = str(self._load_dist_profile_combo.currentData())
        total = float(self._load_dist_total_mag_spin.value())
        norm = hypot(vx, vy)
        if load_type != "pressure" or direction_mode == "vector":
            if norm <= 1e-12:
                self._show_error(
                    self._ui("载荷失败", "Load failed"),
                    self._ui("方向向量不能为零，请输入有效方向。", "Direction vector cannot be zero."),
                )
                return
        if load_type not in {"distributed", "pressure"}:
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui("该面板仅支持边线载或边压力。", "This panel only supports edge line load or edge pressure."),
            )
            return

        if self._current_load_bc_target_mode() == "geometry":
            if self._current_load_bc_target_set_id() is None and self._selected_geometry_edge_ids:
                preferred_set_id = self._preferred_geometry_set_id_for_pick("edge")
                self._upsert_geometry_set_from_geometry_ids(
                    entity_type="edge",
                    geometry_ids=self._selected_geometry_edge_ids,
                    preferred_name=self._ui("几何边集合", "Geometry Edge Set"),
                    preferred_set_id=preferred_set_id,
                )
            set_id = self._current_load_bc_target_set_id()
            target_set = self._scene_project.geometry_sets.get(set_id or "")
            if target_set is None:
                if self._scene_project.geometry_edges:
                    self._start_pick_distributed_edge_target()
                    return
                self._show_error(
                    self._ui("载荷失败", "Load failed"),
                    self._ui("分布力需要边集。请先点击“几何选边”并在画布中选择目标边。", "Distributed load needs an edge set. Click Pick Geometry Edge and select a target edge first."),
                )
                return
            if target_set.entity_type != "edge":
                if self._scene_project.geometry_edges:
                    self._start_pick_distributed_edge_target()
                    return
                self._show_error(
                    self._ui("载荷失败", "Load failed"),
                    self._ui("当前选择的是点集，分布力需要边集。请点击“几何选边”。", "The current target is a point set; distributed load needs an edge set. Use Pick Geometry Edge."),
                )
                return
            load_id = self._next_load_definition_id()
            self._scene_project.load_definitions[load_id] = LoadDefinition(
                id=load_id,
                name=self._consume_pending_load_name("pressure" if load_type == "pressure" else "distributed"),
                target_set_id=target_set.id,
                target_entity_type="edge",
                load_type="pressure" if load_type == "pressure" else "distributed",
                vector_x=vx,
                vector_y=vy,
                magnitude=total,
                profile="linear" if profile == "linear" else "uniform",
                step=self._consume_pending_load_step(),
                direction_mode="normal" if load_type == "pressure" and direction_mode not in {"vector", "normal", "reverse_normal"} else direction_mode,
            )
            unresolved = self._resolve_scene_load_bc_definitions_to_model() if self._model.mesh.nodes else []
            self._sync_load_bc_geometry_overlay()
            self._refresh_load_bc_managers()
            self._sync_editor_tables_from_model()
            self._update_model_tree()
            self._update_status_labels()
            self._mark_results_stale("Geometry distributed load definition updated")
            if unresolved:
                self._load_bc_set_hint.setText(
                    self._ui(
                        "边载荷已保存到几何边；当前网格尚未解析该边，生成保点网格后会按边段长度展开为等效节点力。",
                        "Edge load saved on geometry; it will resolve to length-consistent nodal loads after meshing.",
                    )
                )
            else:
                self._load_bc_set_hint.setText(
                    self._ui(
                        f"边载荷已保存到几何边集 {target_set.name}。",
                        f"Edge load saved on geometry edge set {target_set.name}.",
                    )
                )
            self._log(
                self._ui(
                    f"几何边载荷已创建: 类型={load_type}, 集合={target_set.name}, 方向=({vx:.4g},{vy:.4g}), 大小={total:.6g}, 轮廓={profile}。",
                    f"Geometry edge load created: type={load_type}, set={target_set.name}, dir=({vx:.4g},{vy:.4g}), mag={total:.6g}, profile={profile}.",
                )
            )
            return

        if load_type == "pressure" and direction_mode != "vector":
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui(
                    "网格目标模式下无法自动判断边法向；请切换到几何目标，或将方向模式改为“自定义向量”。",
                    "Mesh-target mode cannot infer edge normals. Switch to Geometry Target or use Custom Vector direction.",
                ),
            )
            return

        target_nodes = []
        set_nodes = self._selected_load_bc_set_node_ids()
        if self._current_load_bc_target_mode() == "geometry" and not set_nodes:
            if self._current_load_bc_target_set_id() is None:
                self._show_error(
                    self._ui("载荷失败", "Load failed"),
                    self._ui("几何模式下请先创建/选择几何边集合。", "In geometry mode, please create/select a geometry edge set first."),
                )
                return
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui("几何边目标尚未解析到网格节点。请先生成网格并确保边界节点可解析。", "Geometry edge target is unresolved to mesh nodes. Generate mesh first."),
            )
            return
        if set_nodes:
            node_by_id = {int(node.id): node for node in self._model.mesh.nodes}
            for node_id in set_nodes:
                node = node_by_id.get(int(node_id))
                if node is not None:
                    target_nodes.append(node)
        else:
            for node in self._model.mesh.nodes:
                value = node.x if axis == "x" else node.y
                if abs(value - coord) <= tol:
                    target_nodes.append(node)
        if not target_nodes:
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui("未找到匹配边界节点。请检查轴向、坐标和容差。", "No edge nodes matched for distributed load."),
            )
            return

        if profile == "uniform":
            weights = [1.0 for _ in target_nodes]
        else:
            reference = [node.y if axis == "x" else node.x for node in target_nodes]
            rmin = min(reference)
            rmax = max(reference)
            span = max(rmax - rmin, 1e-9)
            weights = [0.2 + (value - rmin) / span for value in reference]
        weight_sum = sum(weights)
        if abs(weight_sum) <= 1e-12:
            self._show_error(self._ui("载荷失败", "Load failed"), self._ui("分布载荷权重无效。", "Invalid weight definition for distributed load."))
            return

        dir_x = vx / norm
        dir_y = vy / norm
        total_fx = total * dir_x
        total_fy = total * dir_y

        for node, weight in zip(target_nodes, weights):
            ratio = weight / weight_sum
            fx = total_fx * ratio
            fy = total_fy * ratio
            if abs(fx) > 1e-12:
                self._model.loads.append(Load(node_id=node.id, dof="fx", value=fx))
            if abs(fy) > 1e-12:
                self._model.loads.append(Load(node_id=node.id, dof="fy", value=fy))
        self._set_model(self._model)
        self._upsert_geometry_set_from_node_ids(
            entity_type="edge",
            node_ids=[int(node.id) for node in target_nodes],
            preferred_name=self._ui("边载荷集合", "Edge Load Set"),
        )
        self._mark_results_stale("Distributed load updated")
        self._log(
            self._ui(
                f"分布力已添加: 节点数={len(target_nodes)}, 轴={axis}, 坐标={coord:.4f}, 方向=({vx:.4g},{vy:.4g}), 总大小={total:.6g}, 轮廓={profile}。",
                f"Distributed load added: nodes={len(target_nodes)}, axis={axis}, coord={coord:.4f}, dir=({vx:.4g},{vy:.4g}), total={total:.6g}, profile={profile}.",
            )
        )

    def _current_region_or_component_target_set(self) -> GeometrySetDef | None:
        target_set = self._current_load_bc_target_set()
        if target_set is not None and target_set.entity_type in {"region", "component"}:
            return target_set
        region_ids = self._selected_scene_region_ids_for_component_assignment()
        if region_ids:
            set_id = self._upsert_geometry_set_from_entities(
                entity_type="region",
                entity_ids=region_ids,
                preferred_name=self._ui("区域集合", "Region Set"),
                binding_mode="geometry",
            )
            return self._scene_project.geometry_sets.get(set_id or "")
        component_id = self._component_manager_selected_id() or self._scene_selection.active_component_id
        if component_id is not None and component_id in self._scene_project.components:
            set_id = self._upsert_geometry_set_from_entities(
                entity_type="component",
                entity_ids=[component_id],
                preferred_name=self._ui("组件集合", "Component Set"),
                binding_mode="geometry",
            )
            return self._scene_project.geometry_sets.get(set_id or "")
        return None

    def _add_area_load_definition(self, *, load_type: str) -> None:
        if self._model is None:
            self._create_empty_model()
        if self._model is None:
            return
        if self._current_load_bc_target_mode() != "geometry":
            idx = self._load_bc_target_mode_combo.findData("geometry")
            if idx >= 0:
                self._load_bc_target_mode_combo.setCurrentIndex(idx)
        target_set = self._current_region_or_component_target_set()
        if target_set is None:
            self._start_pick_load_bc_region_target()
            self._load_bc_set_hint.setText(
                self._ui(
                    "区域体力/自重需要区域集或组件集：请先画布选区域并创建区域集合，或在 Assembly 选择组件。",
                    "Body force/gravity needs a region or component set. Pick regions and capture a set, or select a component in Assembly.",
                )
            )
            return
        if target_set.entity_type not in {"region", "component"}:
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui("当前目标不是区域集/组件集，不能用于区域体力或自重。", "Current target is not a region/component set and cannot be used for body force/gravity."),
            )
            return
        vx = float(self._region_load_vec_x.value())
        vy = float(self._region_load_vec_y.value())
        if load_type == "gravity" and hypot(vx, vy) <= 1e-12:
            vx, vy = 0.0, -1.0
        if hypot(vx, vy) <= 1e-12:
            self._show_error(
                self._ui("载荷失败", "Load failed"),
                self._ui("区域体力方向不能为零。", "Body-force direction cannot be zero."),
            )
            return
        intensity = float(self._region_load_intensity_spin.value())
        load_id = self._next_load_definition_id()
        self._scene_project.load_definitions[load_id] = LoadDefinition(
            id=load_id,
            name=(
                self._consume_pending_load_name("gravity")
                if load_type == "gravity"
                else self._consume_pending_load_name("body")
            ),
            target_set_id=target_set.id,
            target_entity_type=target_set.entity_type,
            load_type="gravity" if load_type == "gravity" else "body",
            vector_x=vx,
            vector_y=vy,
            magnitude=intensity,
            profile="uniform",
            step=self._consume_pending_load_step(),
            direction_mode="gravity" if load_type == "gravity" else "vector",
        )
        unresolved = self._resolve_scene_load_bc_definitions_to_model() if self._model.mesh.nodes else []
        self._sync_load_bc_geometry_overlay()
        self._refresh_load_bc_managers()
        self._sync_editor_tables_from_model()
        self._update_model_tree()
        self._update_status_labels()
        self._mark_results_stale("Geometry area load definition updated")
        self._load_bc_set_hint.setText(
            self._ui(
                "区域载荷已保存；若当前未解析到单元，会在网格生成/求解前再次解析。",
                "Area load saved; if currently unresolved, it will be retried before solve/after meshing.",
            )
            if unresolved
            else self._ui(
                f"区域载荷已保存到 {target_set.name}。",
                f"Area load saved on {target_set.name}.",
            )
        )

    def _add_region_body_force(self) -> None:
        self._add_area_load_definition(load_type="body")

    def _add_region_gravity(self) -> None:
        self._add_area_load_definition(load_type="gravity")

    def _apply_constraint(self) -> None:
        if self._model is None:
            self._create_empty_model()
        if self._model is None:
            return
        target_type = str(self._bc_target_combo.currentData())
        direction = str(self._bc_dir_combo.currentData())
        value = float(self._bc_value_spin.value())

        if direction == "rot":
            self._show_error("Constraint note", "Rotation dof is reserved for future 2D beam/shell extension.")
            return

        if self._current_load_bc_target_mode() == "geometry":
            if self._current_load_bc_target_set_id() is None:
                if target_type == "point" and self._selected_geometry_point_ids:
                    preferred_set_id = self._preferred_geometry_set_id_for_pick("point")
                    self._upsert_geometry_set_from_geometry_ids(
                        entity_type="point",
                        geometry_ids=self._selected_geometry_point_ids,
                        preferred_name=self._ui("几何点集合", "Geometry Point Set"),
                        preferred_set_id=preferred_set_id,
                    )
                elif target_type == "edge" and self._selected_geometry_edge_ids:
                    preferred_set_id = self._preferred_geometry_set_id_for_pick("edge")
                    self._upsert_geometry_set_from_geometry_ids(
                        entity_type="edge",
                        geometry_ids=self._selected_geometry_edge_ids,
                        preferred_name=self._ui("几何边集合", "Geometry Edge Set"),
                        preferred_set_id=preferred_set_id,
                    )
            set_id = self._current_load_bc_target_set_id()
            target_set = self._scene_project.geometry_sets.get(set_id or "")
            if target_set is None:
                self._show_error(
                    self._ui("约束失败", "Constraint failed"),
                    self._ui("几何约束需要点集或边集。请先点击“几何选点/几何选边”选择目标。", "Geometry BC needs a point or edge set. Pick a geometry point/edge first."),
                )
                return
            if target_set.entity_type not in {"point", "edge"}:
                self._show_error(
                    self._ui("约束失败", "Constraint failed"),
                    self._ui("当前集合不是点集或边集，暂不能用于位移约束。", "The selected set is not a point or edge set and cannot be used for displacement BCs yet."),
                )
                return
            bc_id = self._next_boundary_definition_id()
            self._scene_project.boundary_definitions[bc_id] = BoundaryDefinition(
                id=bc_id,
                name=self._consume_pending_boundary_name(),
                target_set_id=target_set.id,
                target_entity_type="point" if target_set.entity_type == "point" else "edge",
                direction="x" if direction == "x" else "y" if direction == "y" else "xy",
                value=value,
                step=self._consume_pending_boundary_step(),
            )
            unresolved = self._resolve_scene_load_bc_definitions_to_model() if self._model.mesh.nodes else []
            self._sync_load_bc_geometry_overlay()
            self._refresh_load_bc_managers()
            self._sync_editor_tables_from_model()
            self._update_model_tree()
            self._update_status_labels()
            self._mark_results_stale("Geometry boundary definition updated")
            if unresolved:
                self._load_bc_set_hint.setText(
                    self._ui(
                        "约束已保存为几何边界条件；生成保点网格后会自动展开到节点。",
                        "Boundary condition saved on geometry and will resolve to nodes after meshing.",
                    )
                )
            else:
                self._load_bc_set_hint.setText(
                    self._ui(
                        f"约束已保存到几何集合 {target_set.name}。",
                        f"Boundary condition saved on geometry set {target_set.name}.",
                    )
                )
            self._log(
                self._ui(
                    f"几何约束已创建: 集合={target_set.name}, 自由度={direction}, 值={value:.6g}。",
                    f"Geometry BC created: set={target_set.name}, dof={direction}, value={value:.6g}.",
                )
            )
            return

        target_node_ids: list[int] = []
        set_nodes = self._selected_load_bc_set_node_ids()
        if self._current_load_bc_target_mode() == "geometry" and not set_nodes:
            if self._current_load_bc_target_set_id() is None:
                self._show_error(
                    self._ui("约束失败", "Constraint failed"),
                    self._ui("几何模式下请先创建/选择几何点集或边集。", "In geometry mode, please create/select a geometry point/edge set first."),
                )
                return
            self._show_error(
                self._ui("约束失败", "Constraint failed"),
                self._ui("几何目标尚未解析到网格节点。请先生成网格并确保几何目标被保留。", "Geometry target is unresolved to mesh nodes. Generate mesh and preserve target first."),
            )
            return
        if set_nodes:
            target_node_ids = list(set_nodes)
        elif target_type == "point":
            node_id = int(self._bc_point_node_spin.value())
            if self._get_node_by_id(node_id) is None:
                self._show_error("Constraint failed", f"Node {node_id} does not exist.")
                return
            target_node_ids = [node_id]
        elif target_type == "edge":
            axis = str(self._bc_edge_axis_combo.currentData())
            coord = float(self._bc_edge_coord_spin.value())
            tol = float(self._bc_edge_tol_spin.value())
            for node in self._model.mesh.nodes:
                value_axis = node.x if axis == "x" else node.y
                if abs(value_axis - coord) <= tol:
                    target_node_ids.append(node.id)
        else:
            target_node_ids = [node.id for node in self._model.mesh.nodes]

        if not target_node_ids:
            self._show_error("Constraint failed", "No target nodes selected by current options.")
            return

        if direction == "x":
            dofs = ("ux",)
        elif direction == "y":
            dofs = ("uy",)
        else:
            dofs = ("ux", "uy")

        bc_by_key: dict[tuple[int, str], BoundaryCondition] = {
            (bc.node_id, bc.dof): bc
            for bc in self._model.boundary_conditions
        }
        for node_id in target_node_ids:
            for dof in dofs:
                bc_by_key[(node_id, dof)] = BoundaryCondition(node_id=node_id, dof=dof, value=value)
        self._model.boundary_conditions = sorted(
            bc_by_key.values(),
            key=lambda item: (item.node_id, item.dof),
        )
        self._set_model(self._model)
        self._upsert_geometry_set_from_node_ids(
            entity_type="point" if len(target_node_ids) == 1 else "edge",
            node_ids=target_node_ids,
            preferred_name=self._ui("约束集合", "Constraint Set"),
        )
        self._mark_results_stale("Boundary condition updated")
        self._log(
            f"Constraint applied: target={target_type}, nodes={len(target_node_ids)}, dof={direction}, value={value:.6g}."
        )

    def _generate_mesh_from_controls(self) -> None:
        algorithm = str(self._mesh_algo_combo.currentData())
        if algorithm == "import":
            self._import_model_file()
            return
        self._scene_project.mesh_backend = self._mesh_backend_name()
        self._scene_project.global_mesh_size = float(self._mesh_global_seed_spin.value())
        if self._part_sketch_points or self._scene_project.regions:
            self._sync_scene_from_sketch_geometry()
            if self._scene_project.regions:
                self._run_mesh_backend_operation(operation="generate")
                return
        if self._requires_constrained_geometry_mesh() and algorithm in {"structured", "delaunay"}:
            try:
                model = self._build_model_from_scene_constrained()
            except Exception as exc:
                self._show_error("Mesh generation failed", str(exc))
                return
            self._clear_mapping_state()
            self._set_model(model)
            mapped = self._apply_geometry_material_rules_to_mesh()
            if mapped > 0:
                self._set_model(self._model)
            self._log(
                "Geometry-preserving targets detected; switched to constrained Delaunay mesh generation: "
                f"nodes={len(model.mesh.nodes)}, elements={len(model.mesh.elements)}, mapped_elements={mapped}."
            )
            return
        if algorithm == "github_ear":
            if len(self._part_sketch_points) < 3:
                self._show_error(
                    "Mesh generation failed",
                    "GitHub Ear-Clipping mode requires at least 3 sketch points in Part section.",
                )
                return
            use_constrained = self._requires_constrained_geometry_mesh()
            try:
                if use_constrained:
                    model = self._build_model_from_scene_constrained()
                else:
                    model = self._build_model_from_polygon_points(list(self._part_sketch_points))
            except Exception as exc:
                self._show_error("Mesh generation failed", str(exc))
                return
            self._clear_mapping_state()
            self._set_model(model)
            mapped = self._apply_geometry_material_rules_to_mesh()
            if mapped > 0:
                self._set_model(self._model)
            self._log(
                f"Mesh generated using {'constrained Delaunay' if use_constrained else 'GitHub ear-clipping'} algorithm from sketch: "
                f"nodes={len(model.mesh.nodes)}, elements={len(model.mesh.elements)}, mapped_elements={mapped}."
            )
            return

        width = float(self._mesh_width_spin.value())
        height = float(self._mesh_height_spin.value())
        global_seed = float(self._mesh_global_seed_spin.value())
        local_seed = float(self._mesh_local_seed_spin.value())
        use_local = bool(self._mesh_use_local_seed.isChecked())
        xmin = float(self._mesh_local_xmin_spin.value())
        xmax = float(self._mesh_local_xmax_spin.value())
        ymin = float(self._mesh_local_ymin_spin.value())
        ymax = float(self._mesh_local_ymax_spin.value())

        coords_x = self._axis_seed_coords(
            width,
            global_seed,
            local_seed=local_seed,
            use_local=use_local,
            local_min=xmin,
            local_max=xmax,
        )
        coords_y = self._axis_seed_coords(
            height,
            global_seed,
            local_seed=local_seed,
            use_local=use_local,
            local_min=ymin,
            local_max=ymax,
        )

        if self._model and self._model.materials:
            base_material = copy.deepcopy(sorted(self._model.materials, key=lambda item: item.id)[0])
        else:
            base_material = Material(
                id=1,
                young_modulus=float(self._material_e_spin.value()),
                poisson_ratio=float(self._material_nu_spin.value()),
                plane_stress=bool(self._material_plane_stress.isChecked()),
            )

        if algorithm == "structured":
            model = self._build_grid_model(coords_x, coords_y, base_material)
            self._clear_mapping_state()
            self._set_model(model)
            mapped = self._apply_geometry_material_rules_to_mesh()
            if mapped > 0:
                self._set_model(self._model)
            self._log(
                f"Mesh generated (structured): nx={len(coords_x) - 1}, ny={len(coords_y) - 1}, "
                f"elements={len(model.mesh.elements)}, mapped_elements={mapped}."
            )
            return

        try:
            from scipy.spatial import Delaunay
        except Exception as exc:  # pragma: no cover
            self._show_error("Mesh failed", f"Delaunay backend unavailable: {exc}")
            return

        points = [(x, y) for y in coords_y for x in coords_x]
        tri = Delaunay(points)
        nodes = [Node(id=i + 1, x=float(x), y=float(y)) for i, (x, y) in enumerate(points)]
        elements: list[Element] = []
        for simplex in tri.simplices:
            n1 = int(simplex[0]) + 1
            n2 = int(simplex[1]) + 1
            n3 = int(simplex[2]) + 1
            p1 = points[n1 - 1]
            p2 = points[n2 - 1]
            p3 = points[n3 - 1]
            signed_two_area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            if abs(signed_two_area) <= 1e-12:
                continue
            if signed_two_area < 0.0:
                n2, n3 = n3, n2
            elements.append(
                Element(
                    id=len(elements) + 1,
                    type="T3",
                    connectivity=[n1, n2, n3],
                    material_id=base_material.id,
                )
            )
        model = Model(mesh=Mesh(nodes=nodes, elements=elements), materials=[base_material])
        self._clear_mapping_state()
        self._set_model(model)
        mapped = self._apply_geometry_material_rules_to_mesh()
        if mapped > 0:
            self._set_model(self._model)
        self._log(
            f"Mesh generated (Delaunay): points={len(points)}, elements={len(elements)}, "
            f"local_seed={'on' if use_local else 'off'}, mapped_elements={mapped}."
        )

    def _submit_job(self) -> None:
        job_name = self._job_name_edit.text().strip() or "Job"
        self._log(f"Submitting job: {job_name}")
        self._job_status_label.setText(f"Running: {job_name}")
        self._solve()
        if self._result is not None and not self._results_stale:
            self._job_status_label.setText(f"Completed: {job_name}")
            self._activate_workflow("visualization")

    def _set_contour_variable(self, variable: str) -> None:
        index = self._contour_combo.findData(variable)
        if index < 0:
            return
        self._contour_combo.setCurrentIndex(index)
        self._mesh_canvas.set_contour_variable(variable)
        self._side_tabs.setCurrentWidget(self._view_tab)

    def _set_canvas_tool(self, tool: str) -> None:
        normalized = str(tool).strip().lower()
        if normalized not in {"select", "add_node", "add_element", "measure", "measure_angle", "draw_rect", "draw_ellipse", "draw_circle", "pick_geometry"}:
            normalized = "select"
        idx = self._editor_canvas_tool_combo.findData(normalized)
        if idx >= 0:
            if self._editor_canvas_tool_combo.currentIndex() != idx:
                self._editor_canvas_tool_combo.setCurrentIndex(idx)
        self._mesh_canvas.set_edit_tool(normalized)

    def _activate_measure_tool(self) -> None:
        self._pending_pick_context = None
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("measure")
        if hasattr(self, "_part_quick_mode"):
            idx = self._part_quick_mode.findData("length")
            if idx >= 0:
                self._part_quick_mode.setCurrentIndex(idx)
        self._model_measure_result.setText(
            self._ui("测距: 吸附当前草图/历史图形已有点（Shift可自由测量）", "Measure: snap to existing sketch/history points (hold Shift for free-pick)")
        )
        self._log(
            self._ui(
                "测量工具已激活：默认吸附草图点、历史图形点和圆/椭圆中心；按住 Shift 可临时自由测量。",
                "Measure tool activated: snaps to sketch points, history points, and circle/ellipse centers by default; hold Shift for temporary free-pick.",
            )
        )

    def _activate_angle_measure_tool(self) -> None:
        self._pending_pick_context = None
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("measure_angle")
        if hasattr(self, "_part_quick_mode"):
            idx = self._part_quick_mode.findData("angle")
            if idx >= 0:
                self._part_quick_mode.setCurrentIndex(idx)
        self._model_measure_result.setText(
            self._ui("测角: 先选第一条边，再选第二条边（Shift可临时自由测角）", "Measure angle: pick first edge then second edge (hold Shift for free-pick)")
        )
        self._log(
            self._ui(
                "角度测量工具已激活：默认按边顺序选角，第二条边会作为修改侧；按住 Shift 可临时自由测角。",
                "Angle measure tool activated: edge-order based selection by default; second edge is the edited side. Hold Shift for temporary free-pick.",
            )
        )

    def _ensure_sketch_plane_for_drawing(self) -> bool:
        if self._mesh_canvas.is_sketch_plane_active():
            return True
        width = max(float(self._active_sketch_width), 1e-3)
        height = max(float(self._active_sketch_height), 1e-3)
        step = max(float(self._active_sketch_grid_step), 0.05)
        self._mesh_canvas.set_sketch_plane(width, height, step)
        return True

    def _activate_draw_rect_tool(self) -> None:
        self._pending_pick_context = None
        self._activate_workflow("part")
        self._editor_canvas_edit_enable.setChecked(True)
        self._ensure_sketch_plane_for_drawing()
        self._set_canvas_tool("draw_rect")
        self._model_measure_result.setText(
            self._ui("矩形绘制: 拖拽两点定义外包框（Shift锁正方形）", "Rectangle: drag two corners for bounding box (Shift locks square)")
        )
        self._part_dimension_status.setText(
            self._ui("矩形绘制已激活：在画布拖拽创建矩形。", "Rectangle drawing active: drag on canvas to create rectangle.")
        )

    def _activate_draw_ellipse_tool(self) -> None:
        self._pending_pick_context = None
        self._activate_workflow("part")
        self._editor_canvas_edit_enable.setChecked(True)
        self._ensure_sketch_plane_for_drawing()
        self._set_canvas_tool("draw_ellipse")
        self._model_measure_result.setText(
            self._ui("椭圆绘制: 拖拽两点定义外包框（Shift锁圆）", "Ellipse: drag two corners for bounding box (Shift locks circle)")
        )
        self._part_dimension_status.setText(
            self._ui("椭圆绘制已激活：在画布拖拽创建椭圆。", "Ellipse drawing active: drag on canvas to create ellipse.")
        )

    def _activate_draw_circle_tool(self) -> None:
        self._pending_pick_context = None
        self._activate_workflow("part")
        self._editor_canvas_edit_enable.setChecked(True)
        self._ensure_sketch_plane_for_drawing()
        self._set_canvas_tool("draw_circle")
        self._model_measure_result.setText(
            self._ui("圆形绘制: 拖拽定义外包框（自动锁等比例）", "Circle: drag bounding box (ratio locked)")
        )
        self._part_dimension_status.setText(
            self._ui("圆形绘制已激活：在画布拖拽创建圆。", "Circle drawing active: drag on canvas to create circle.")
        )

    def _on_canvas_sketch_primitive_drawn(
        self,
        kind: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        min_x = min(float(x0), float(x1))
        max_x = max(float(x0), float(x1))
        min_y = min(float(y0), float(y1))
        max_y = max(float(y0), float(y1))
        width = max_x - min_x
        height = max_y - min_y
        if width <= 1e-9 or height <= 1e-9:
            return
        self._push_part_undo_snapshot()
        self._archive_current_sketch_to_history()
        self._selected_part_segment = None
        self._selected_part_angle = None
        if kind == "rect":
            points = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)]
            self._part_sketch_points = points
            self._clear_part_curve_hint()
            self._log(self._ui("画布矩形绘制完成。", "Canvas rectangle created."))
        else:
            segments = 48
            cx = 0.5 * (min_x + max_x)
            cy = 0.5 * (min_y + max_y)
            rx = 0.5 * width
            ry = 0.5 * height
            points: list[tuple[float, float]] = []
            for i in range(segments):
                t = 2.0 * 3.141592653589793 * (float(i) / float(segments))
                points.append((cx + rx * cos(t), cy + ry * sin(t)))
            points.append(points[0])
            self._part_sketch_points = points
            hint_kind = "circle" if str(kind).lower() == "circle" else "ellipse"
            self._part_sketch_curve_hint = self._build_curve_hint(hint_kind, min_x, min_y, max_x, max_y)
            self._log(
                self._ui(
                    "画布圆形绘制完成。" if hint_kind == "circle" else "画布椭圆绘制完成。",
                    "Canvas circle created." if hint_kind == "circle" else "Canvas ellipse created.",
                )
            )
        self._sync_part_shape_controls_from_size(width, height)
        self._refresh_part_sketch_table()
        self._show_model_group(self._rock_part_group)

    def _on_measurement_updated(self, distance: float) -> None:
        self._model_measure_result.setText(self._ui(f"测距: L = {distance:.6g}", f"Measure: L = {distance:.6g}"))
        if hasattr(self, "_part_quick_mode"):
            idx = self._part_quick_mode.findData("length")
            if idx >= 0:
                self._part_quick_mode.setCurrentIndex(idx)
            self._part_quick_value.setValue(max(1e-6, distance))
        if hasattr(self, "_part_dimension_status"):
            self._part_dimension_status.setText(
                self._ui(
                    f"当前测距 L = {distance:.6g}，可点击“测距填入长度”用于长度标定。",
                    f"Current measured L = {distance:.6g}. Use 'Use Measured Length' to calibrate.",
                )
            )
        self._log(self._ui(f"测距结果: L = {distance:.6g}", f"Measurement result: L = {distance:.6g}"))

    def _on_measurement_angle_updated(self, angle_deg: float) -> None:
        self._model_measure_result.setText(self._ui(f"测角: θ = {angle_deg:.6g}°", f"Measure angle: θ = {angle_deg:.6g} deg"))
        if hasattr(self, "_part_quick_mode"):
            idx = self._part_quick_mode.findData("angle")
            if idx >= 0:
                self._part_quick_mode.setCurrentIndex(idx)
            self._part_quick_value.setValue(max(1e-6, angle_deg))
        if hasattr(self, "_part_dimension_status"):
            self._part_dimension_status.setText(
                self._ui(
                    f"当前测角 θ = {angle_deg:.6g}°，可直接应用角度标定。",
                    f"Current measured angle θ = {angle_deg:.6g} deg. You can apply angle calibration.",
                )
            )
        self._log(self._ui(f"测角结果: θ = {angle_deg:.6g}°", f"Angle measurement result: θ = {angle_deg:.6g} deg"))

    def _insert_part_rectangle(self) -> None:
        width = max(float(self._part_shape_w.value()), 1e-6) if hasattr(self, "_part_shape_w") else max(float(self._active_sketch_width), 1e-6)
        height = max(float(self._part_shape_h.value()), 1e-6) if hasattr(self, "_part_shape_h") else max(float(self._active_sketch_height), 1e-6)
        self._active_sketch_width = width
        self._active_sketch_height = height
        if not self._mesh_canvas.is_sketch_plane_active():
            plane_w = max(width * 1.6, width + 4.0 * max(float(self._active_sketch_grid_step), 0.05))
            plane_h = max(height * 1.6, height + 4.0 * max(float(self._active_sketch_grid_step), 0.05))
            self._mesh_canvas.set_sketch_plane(plane_w, plane_h, max(float(self._active_sketch_grid_step), 0.05))
        self._push_part_undo_snapshot()
        self._archive_current_sketch_to_history()
        self._selected_part_segment = None
        self._selected_part_angle = None
        ox = 0.5 * max(float(self._active_sketch_grid_step), 0.05)
        oy = 0.5 * max(float(self._active_sketch_grid_step), 0.05)
        self._part_sketch_points = [(ox, oy), (ox + width, oy), (ox + width, oy + height), (ox, oy + height), (ox, oy)]
        self._clear_part_curve_hint()
        self._sync_part_shape_controls_from_size(width, height)
        self._refresh_part_sketch_table()
        self._show_model_group(self._rock_part_group)
        self._log(self._ui("已插入矩形草图。", "Rectangle sketch inserted."))

    def _insert_part_circle(self) -> None:
        width = max(float(self._part_shape_w.value()), 1e-6) if hasattr(self, "_part_shape_w") else max(float(self._active_sketch_width), 1e-6)
        height = max(float(self._part_shape_h.value()), 1e-6) if hasattr(self, "_part_shape_h") else max(float(self._active_sketch_height), 1e-6)
        diameter = max(min(width, height), 1e-6)
        self._active_sketch_width = diameter
        self._active_sketch_height = diameter
        if not self._mesh_canvas.is_sketch_plane_active():
            plane_w = max(diameter * 1.6, diameter + 4.0 * max(float(self._active_sketch_grid_step), 0.05))
            plane_h = max(diameter * 1.6, diameter + 4.0 * max(float(self._active_sketch_grid_step), 0.05))
            self._mesh_canvas.set_sketch_plane(
                plane_w,
                plane_h,
                max(float(self._active_sketch_grid_step), 0.05),
            )
        self._push_part_undo_snapshot()
        self._archive_current_sketch_to_history()
        self._selected_part_segment = None
        self._selected_part_angle = None
        margin = 0.5 * max(float(self._active_sketch_grid_step), 0.05)
        center_x = margin + 0.5 * diameter
        center_y = margin + 0.5 * diameter
        radius = 0.5 * diameter
        segments = 48
        points: list[tuple[float, float]] = []
        for i in range(segments):
            t = 2.0 * 3.141592653589793 * (float(i) / float(segments))
            points.append((center_x + radius * cos(t), center_y + radius * sin(t)))
        points.append(points[0])
        self._part_sketch_points = points
        self._part_sketch_curve_hint = self._build_curve_hint(
            "circle",
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        )
        self._sync_part_shape_controls_from_size(diameter, diameter)
        self._refresh_part_sketch_table()
        self._show_model_group(self._rock_part_group)
        self._log(self._ui("已插入圆形草图。", "Circle sketch inserted."))

    def _insert_part_ellipse(self) -> None:
        width = max(float(self._part_shape_w.value()), 1e-6) if hasattr(self, "_part_shape_w") else max(float(self._active_sketch_width), 1e-6)
        height = max(float(self._part_shape_h.value()), 1e-6) if hasattr(self, "_part_shape_h") else max(float(self._active_sketch_height), 1e-6)
        self._active_sketch_width = width
        self._active_sketch_height = height
        if not self._mesh_canvas.is_sketch_plane_active():
            plane_w = max(width * 1.6, width + 4.0 * max(float(self._active_sketch_grid_step), 0.05))
            plane_h = max(height * 1.6, height + 4.0 * max(float(self._active_sketch_grid_step), 0.05))
            self._mesh_canvas.set_sketch_plane(
                plane_w,
                plane_h,
                max(float(self._active_sketch_grid_step), 0.05),
            )
        self._push_part_undo_snapshot()
        self._archive_current_sketch_to_history()
        self._selected_part_segment = None
        self._selected_part_angle = None
        margin = 0.5 * max(float(self._active_sketch_grid_step), 0.05)
        center_x = margin + 0.5 * width
        center_y = margin + 0.5 * height
        rx = 0.5 * width
        ry = 0.5 * height
        segments = 48
        points: list[tuple[float, float]] = []
        for i in range(segments):
            t = 2.0 * 3.141592653589793 * (float(i) / float(segments))
            points.append((center_x + rx * cos(t), center_y + ry * sin(t)))
        points.append(points[0])
        self._part_sketch_points = points
        self._part_sketch_curve_hint = self._build_curve_hint(
            "ellipse",
            center_x - rx,
            center_y - ry,
            center_x + rx,
            center_y + ry,
        )
        self._sync_part_shape_controls_from_size(width, height)
        self._refresh_part_sketch_table()
        self._show_model_group(self._rock_part_group)
        self._log(self._ui("已插入椭圆草图。", "Ellipse sketch inserted."))

    def _apply_quick_dimension_value(self) -> None:
        if self._part_quick_mode.currentData() == "angle":
            self._part_angle_target.setValue(float(self._part_quick_value.value()))
            self._apply_part_angle_constraint()
            return
        self._part_length_target.setValue(float(self._part_quick_value.value()))
        self._apply_part_length_constraint()

    def _on_mesh_backend_changed(self, _: int) -> None:
        self._scene_project.mesh_backend = self._mesh_backend_name()
        if self._scene_project.mesh_backend == "gmsh":
            self._mesh_selection_hint.setText(
                self._ui(
                    "Gmsh 后端已启用：几何点/边/区域会作为映射约束返回。",
                    "Gmsh backend enabled: geometry point/edge/region mappings will be returned.",
                )
            )
        else:
            self._update_mesh_selection_hint()

    def _start_pick_mesh_region(self) -> None:
        if self._pending_pick_context in {"mesh_region", "mesh_region_sketch"}:
            self._pending_pick_context = None
            self._update_mesh_selection_hint()
            self._sync_pickable_sketch_faces_to_canvas()
            return
        has_mesh_elements = self._model is not None and bool(self._model.mesh.elements)
        self._selected_face_region_element_ids.clear()
        self._selected_face_region_sketch_keys.clear()
        self._mesh_canvas.set_highlighted_elements(set())
        self._mesh_canvas.set_highlighted_sketch_faces([])
        if has_mesh_elements:
            self._pending_pick_context = "mesh_region"
            self._mesh_selection_hint.setText(
                self._ui(
                    "等待在画布中点选连通区域（支持连续多选）...",
                    "Waiting for connected-region picks (continuous multi-select enabled)...",
                )
            )
        else:
            self._sync_scene_from_sketch_geometry()
            sketch_regions = self._collect_sketch_face_regions()
            if not sketch_regions:
                self._show_error(
                    self._ui("选择失败", "Pick failed"),
                    self._ui("当前没有可选网格区域或封闭草图区域。", "No mesh regions or closed sketch faces available."),
                )
                return
            self._pending_pick_context = "mesh_region_sketch"
            self._mesh_selection_hint.setText(
                self._ui(
                    "等待在画布中点选草图区域（支持连续多选）...",
                    "Waiting for sketch-region picks (continuous multi-select enabled)...",
                )
            )
            self._sync_pickable_sketch_faces_to_canvas()
            self._log(
                self._ui(
                    f"网格区域选区：已拆分 {len(sketch_regions)} 个原子子区域，请继续点选；Esc 结束。",
                    f"Mesh-region pick: {len(sketch_regions)} atomic sketch regions are available; keep clicking to add/remove and press Esc to finish.",
                )
            )
        if has_mesh_elements:
            self._sync_pickable_sketch_faces_to_canvas()
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("select")

    def _start_pick_mesh_edge_seed(self) -> None:
        self._sync_scene_from_sketch_geometry()
        if not self._scene_project.geometry_edges:
            self._show_error(
                self._ui("边选择失败", "Edge selection failed"),
                self._ui("当前没有可选几何边。请先绘制闭合草图。", "No geometry edges are available. Draw a closed sketch first."),
            )
            return
        if self._pending_pick_context == "mesh_edge_seed":
            self._pending_pick_context = None
            self._selected_geometry_edge_ids.clear()
            self._update_mesh_selection_hint()
            self._sync_load_bc_geometry_overlay()
            return
        self._pending_pick_context = "mesh_edge_seed"
        self._selected_geometry_edge_ids.clear()
        self._update_mesh_selection_hint()
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("pick_geometry")

    def _apply_mesh_size_to_selected_regions(self) -> None:
        region_ids = self._current_mesh_selection_region_ids()
        if not region_ids:
            self._show_error(
                self._ui("区域种子失败", "Region seed failed"),
                self._ui("请先选择区域。", "Pick one or more regions first."),
            )
            return
        value = float(self._mesh_region_seed_spin.value())
        updated = 0
        for region_id in region_ids:
            region = self._scene_project.regions.get(region_id)
            if region is None:
                continue
            region.mesh_size = value
            updated += 1
        self._update_mesh_selection_hint()
        self._log(f"Applied region seed {value:.6g} to {updated} region(s).")

    def _apply_mesh_size_to_selected_edges(self) -> None:
        if not self._selected_geometry_edge_ids:
            self._show_error(
                self._ui("边种子失败", "Edge seed failed"),
                self._ui("请先选择边。", "Pick one or more geometry edges first."),
            )
            return
        value = float(self._mesh_edge_seed_spin.value())
        for edge_id in self._selected_geometry_edge_ids:
            self._scene_project.edge_mesh_sizes[str(edge_id)] = value
        self._update_mesh_selection_hint()
        self._sync_load_bc_geometry_overlay()
        self._log(f"Applied edge seed {value:.6g} to {len(self._selected_geometry_edge_ids)} edge(s).")

    def _generate_selected_region_mesh(self) -> None:
        region_ids = self._current_mesh_selection_region_ids()
        if not region_ids:
            self._show_error(
                self._ui("局部生成失败", "Selected mesh generation failed"),
                self._ui("请先选择区域。", "Pick one or more regions first."),
            )
            return
        operation = "remesh" if self._model is not None and self._model.mesh.elements else "generate"
        self._run_mesh_backend_operation(operation=operation, target_region_ids=region_ids)

    def _remesh_selected_region_mesh(self) -> None:
        region_ids = self._current_mesh_selection_region_ids()
        if not region_ids:
            self._show_error(
                self._ui("局部重网格失败", "Local remesh failed"),
                self._ui("请先选择区域。", "Pick one or more regions first."),
            )
            return
        self._run_mesh_backend_operation(operation="remesh", target_region_ids=region_ids)

    def _delete_selected_mesh_elements(self) -> None:
        target_ids = self._current_mesh_selection_element_ids()
        if not target_ids:
            region_ids = self._current_mesh_selection_region_ids()
            for region_id in region_ids:
                target_ids.update(int(item) for item in self._scene_mesh_state.region_to_element_ids.get(region_id, []))
        if not target_ids:
            self._show_error(
                self._ui("删除网格失败", "Delete mesh failed"),
                self._ui("请先选择要删除的单元或区域。", "Select mesh elements or a region first."),
            )
            return
        self._run_mesh_backend_operation(operation="delete", target_element_ids=target_ids)

    def _check_mesh_quality(self) -> None:
        if self._model is None or not self._model.mesh.elements:
            self._show_error(
                self._ui("质量检查失败", "Quality check failed"),
                self._ui("当前没有可检查的网格。", "No mesh is available for quality checks."),
            )
            return
        report = evaluate_t3_mesh_quality(self._model.mesh)
        self._last_mesh_quality_report = report
        self._scene_mesh_state.quality_report = report
        self._refresh_mesh_quality_summary()
        self._log(report.summary)

    def _toggle_bad_mesh_highlight(self) -> None:
        self._mesh_bad_elements_visible = not self._mesh_bad_elements_visible
        self._refresh_mesh_quality_summary()
        self._btn_mesh_toggle_bad.setText(
            self._ui("隐藏坏单元" if self._mesh_bad_elements_visible else "高亮坏单元", "Hide Bad" if self._mesh_bad_elements_visible else "Highlight Bad")
        )

    def _start_pick_face_region(self) -> None:
        if self._pending_pick_context in {"material_face", "material_face_sketch"}:
            self._pending_pick_context = None
            self._material_face_hint.setText(self._tr("rock.material.pick_face.exit", "已退出连续选区"))
            self._sync_pickable_sketch_faces_to_canvas()
            return
        has_mesh_elements = self._model is not None and bool(self._model.mesh.elements)
        if has_mesh_elements:
            self._pending_pick_context = "material_face"
            self._selected_face_region_element_ids.clear()
            self._selected_face_region_sketch_keys.clear()
            self._mesh_canvas.set_highlighted_elements(set())
            self._mesh_canvas.set_highlighted_sketch_faces([])
            self._material_face_hint.setText(
                self._ui("等待在画布中点选区域（支持连续多选）...", "Waiting for canvas picks (continuous multi-select enabled)...")
            )
            self._log(
                self._ui(
                    "请选择一个单元，系统将自动选择连通区域；可连续点选多个区域，Esc 结束。",
                    "Pick an element and a connected region will be selected; keep clicking to add/remove and press Esc to finish.",
                )
            )
        else:
            sketch_regions = self._collect_sketch_face_regions()
            if not sketch_regions:
                self._show_error(
                    self._ui("选择失败", "Pick failed"),
                    self._ui("当前没有可选单元或封闭草图区域。请先生成网格或绘制闭合草图。", "No mesh elements or closed sketch faces available."),
                )
                return
            self._pending_pick_context = "material_face_sketch"
            self._selected_face_region_element_ids.clear()
            self._selected_face_region_sketch_keys.clear()
            self._mesh_canvas.set_highlighted_elements(set())
            self._mesh_canvas.set_highlighted_sketch_faces([])
            self._material_face_hint.setText(
                self._ui("等待在画布中点选草图面区域（支持连续多选）...", "Waiting for sketch-face picks (continuous multi-select enabled)...")
            )
            self._sync_pickable_sketch_faces_to_canvas()
            self._log(
                self._ui(
                    f"请选择草图封闭面。当前已拆分 {len(sketch_regions)} 个原子子区域；可连续点选多个区域，Esc 结束。",
                    f"Pick closed sketch faces. {len(sketch_regions)} atomic sketch regions are available; keep clicking to add/remove and press Esc to finish.",
                )
            )
        if has_mesh_elements:
            self._sync_pickable_sketch_faces_to_canvas()
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("select")

    def _ensure_nodes_for_point_pick(self, *, purpose: str) -> bool:
        if self._model is not None and self._model.mesh.nodes:
            return True
        if len(self._part_sketch_points) >= 3:
            answer = QMessageBox.question(
                self,
                self._ui("当前没有可选节点", "No selectable nodes"),
                self._ui(
                    "当前模型还没有网格节点。是否先从当前草图自动生成部件，然后继续选点？",
                    "Current model has no mesh nodes. Build part from current sketch first, then continue picking?",
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self._build_part_from_sketch()
                if self._model is not None and self._model.mesh.nodes:
                    return True
        purpose_text = self._ui("集中力选点" if purpose == "load" else "约束选点", "load/BC point picking")
        self._show_error(
            self._ui("选择失败", "Pick failed"),
            self._ui(
                f"{purpose_text}需要先有网格节点。请先在 Part 里封闭草图生成部件，或在 Mesh 里生成/导入网格。",
                f"{purpose_text} needs mesh nodes first. Build part from closed sketch, or generate/import mesh in Mesh step.",
            ),
        )
        return False

    def _start_pick_load_point(self) -> None:
        if self._current_load_bc_target_mode() == "geometry":
            self._start_pick_geometry_point_target()
            return
        if not self._ensure_nodes_for_point_pick(purpose="load"):
            return
        self._pending_pick_context = "load_point"
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("select")
        self._mesh_canvas.set_highlighted_nodes(set())
        self._log(self._ui("请在画布上选择集中力作用点。", "Pick a node for concentrated load."))

    def _start_pick_bc_point(self) -> None:
        if self._current_load_bc_target_mode() == "geometry":
            self._start_pick_geometry_point_target()
            return
        if not self._ensure_nodes_for_point_pick(purpose="bc"):
            return
        self._pending_pick_context = "bc_point"
        self._editor_canvas_edit_enable.setChecked(True)
        self._set_canvas_tool("select")
        self._mesh_canvas.set_highlighted_nodes(set())
        idx = self._bc_target_combo.findData("point")
        if idx >= 0:
            self._bc_target_combo.setCurrentIndex(idx)
        self._log(self._ui("请在画布上选择约束点。", "Pick a node for point constraint."))

    def _start_pick_bc_edge(self) -> None:
        if hasattr(self, "_load_bc_target_mode_combo"):
            mode_idx = self._load_bc_target_mode_combo.findData("geometry")
            if mode_idx >= 0 and self._load_bc_target_mode_combo.currentIndex() != mode_idx:
                self._load_bc_target_mode_combo.setCurrentIndex(mode_idx)
        edge_idx = self._bc_target_combo.findData("edge")
        if edge_idx >= 0:
            self._bc_target_combo.setCurrentIndex(edge_idx)
        self._start_pick_geometry_edge_target()
        if hasattr(self, "_load_bc_set_hint") and self._pending_pick_context == "geometry_edge_target":
            self._load_bc_set_hint.setText(
                self._ui(
                    "边约束目标需要边集：请在画布上点选一条几何边，选中后再点击“应用约束”或快捷约束。",
                    "Edge BC needs an edge set: click a geometry edge on the canvas, then apply a constraint or preset.",
                )
            )

    def _set_bc_direction_and_value(self, direction: str, value: float = 0.0) -> None:
        idx = self._bc_dir_combo.findData(direction)
        if idx >= 0:
            self._bc_dir_combo.setCurrentIndex(idx)
        self._bc_value_spin.setValue(float(value))

    def _apply_bc_fixed_preset(self) -> None:
        self._set_bc_direction_and_value("xy", 0.0)
        self._apply_constraint()

    def _apply_bc_fix_x_preset(self) -> None:
        self._set_bc_direction_and_value("x", 0.0)
        self._apply_constraint()

    def _apply_bc_fix_y_preset(self) -> None:
        self._set_bc_direction_and_value("y", 0.0)
        self._apply_constraint()

    def _apply_bc_roller_x_preset(self) -> None:
        self._set_bc_direction_and_value("x", 0.0)
        self._apply_constraint()

    def _apply_bc_roller_y_preset(self) -> None:
        self._set_bc_direction_and_value("y", 0.0)
        self._apply_constraint()

    def _collect_connected_element_region(self, seed_element_id: int) -> set[int]:
        if self._model is None:
            return set()
        connectivity_by_eid = {
            element.id: tuple(int(node_id) for node_id in element.connectivity)
            for element in self._model.mesh.elements
            if len(element.connectivity) == 3
        }
        if seed_element_id not in connectivity_by_eid:
            return set()

        edge_to_elements: dict[tuple[int, int], list[int]] = {}
        for eid, conn in connectivity_by_eid.items():
            edges = (
                tuple(sorted((conn[0], conn[1]))),
                tuple(sorted((conn[1], conn[2]))),
                tuple(sorted((conn[2], conn[0]))),
            )
            for edge in edges:
                edge_to_elements.setdefault(edge, []).append(eid)

        neighbors: dict[int, set[int]] = {eid: set() for eid in connectivity_by_eid}
        for elem_ids in edge_to_elements.values():
            if len(elem_ids) == 2:
                a, b = elem_ids
                neighbors[a].add(b)
                neighbors[b].add(a)

        visited: set[int] = set()
        queue = [seed_element_id]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for nb in neighbors.get(current, ()):
                if nb not in visited:
                    queue.append(nb)
        return visited

    def _export_canvas_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Canvas Image",
            str(Path.cwd() / "rockfem_view.png"),
            "PNG (*.png)",
        )
        if not path:
            return
        pixmap = self._mesh_canvas.grab()
        if pixmap.save(path):
            self._log(f"Canvas image exported: {path}")
        else:
            self._show_error("Export failed", "Failed to save canvas image.")

    def _compute_result_extremes(
        self,
        *,
        element_ids: set[int] | None = None,
        node_ids: set[int] | None = None,
    ) -> dict[str, float]:
        if self._result is None or self._model is None:
            return {
                "max_displacement": 0.0,
                "max_stress_abs": 0.0,
                "max_principal": 0.0,
                "max_mises": 0.0,
            }
        max_abs_stress = 0.0
        max_principal = 0.0
        max_mises = 0.0
        has_element_sample = False
        for entry in self._result.element_results:
            if element_ids is not None and int(entry.element_id) not in element_ids:
                continue
            sx, sy, txy = (float(item) for item in entry.stress)
            max_abs_stress = max(max_abs_stress, abs(sx), abs(sy), abs(txy))
            radius = sqrt(((sx - sy) * 0.5) ** 2 + txy * txy)
            s1 = 0.5 * (sx + sy) + radius
            mises = sqrt(max(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy, 0.0))
            max_principal = max(max_principal, abs(s1))
            max_mises = max(max_mises, mises)
            has_element_sample = True

        if node_ids is None:
            max_displacement = float(self._result.summary.max_displacement)
        else:
            displacement = self._result.displacements
            max_displacement = 0.0
            for idx, node in enumerate(self._model.mesh.nodes):
                if int(node.id) not in node_ids:
                    continue
                ux = float(displacement[2 * idx])
                uy = float(displacement[2 * idx + 1])
                mag = sqrt(ux * ux + uy * uy)
                if mag > max_displacement:
                    max_displacement = mag

        if not has_element_sample and element_ids is not None:
            max_abs_stress = 0.0
            max_principal = 0.0
            max_mises = 0.0
        return {
            "max_displacement": max_displacement,
            "max_stress_abs": max_abs_stress,
            "max_principal": max_principal,
            "max_mises": max_mises,
        }

    def _update_visualization_metrics(self) -> None:
        if self._result is None:
            if hasattr(self, "_report_metric_label"):
                self._report_metric_label.setText(
                    self._tr(
                        "rock.viz.metrics.empty",
                        "No solved result yet. Submit a job to view max displacement/stress metrics.",
                    )
                )
            if hasattr(self, "_vis_metrics_label"):
                self._vis_metrics_label.setText(
                    self._tr(
                        "rock.viz.metrics.empty",
                        "No solved result yet. Submit a job to view max displacement/stress metrics.",
                    )
                )
            return
        scope_element_ids = self._results_scope_element_ids()
        scope_node_ids = self._results_scope_node_ids()
        ext = self._compute_result_extremes(
            element_ids=scope_element_ids,
            node_ids=scope_node_ids,
        )
        scoped = scope_element_ids is not None or scope_node_ids is not None
        prefix = self._ui("筛选统计", "Filtered") if scoped else self._ui("全局统计", "Global")
        if hasattr(self, "_report_metric_label"):
            self._report_metric_label.setText(
                f"{prefix} | Max displacement: {ext['max_displacement']:.4e} | "
                f"Max |stress|: {ext['max_stress_abs']:.4e} | "
                f"Max principal: {ext['max_principal']:.4e} | "
                f"Mises: {ext['max_mises']:.4e}"
            )
        if hasattr(self, "_vis_metrics_label"):
            self._vis_metrics_label.setText(
                f"{prefix}: |u|max={ext['max_displacement']:.4e}  "
                f"|s|max={ext['max_stress_abs']:.4e}  "
                f"s1,max={ext['max_principal']:.4e}  "
                f"mises,max={ext['max_mises']:.4e}"
            )

    def _generate_engineering_report(self) -> None:
        standard = self._report_standard_combo.currentText()
        context_text = self._report_context_edit.toPlainText().strip()
        ext = self._compute_result_extremes()
        status = summarize_model_state(self._model, backend=self._backend_combo.currentText()) if self._model else {}
        job_name = self._job_name_edit.text().strip() or "RockFEM-Job"
        report = (
            f"# RockFEM 分析报告\n\n"
            f"- 作业名称: {job_name}\n"
            f"- 标准参考: {standard}\n"
            f"- 后端: {self._backend_combo.currentText()}\n\n"
            f"## 模型概况\n"
            f"- 节点数: {status.get('node_count', 0)}\n"
            f"- 单元数: {status.get('element_count', 0)}\n"
            f"- 材料数: {status.get('material_count', 0)}\n"
            f"- 约束数: {status.get('boundary_condition_count', 0)}\n"
            f"- 载荷数: {status.get('load_count', 0)}\n\n"
            f"## 结果摘要\n"
            f"- 最大位移: {ext['max_displacement']:.6e}\n"
            f"- 最大应力绝对值: {ext['max_stress_abs']:.6e}\n"
            f"- 最大主应力: {ext['max_principal']:.6e}\n"
            f"- Mises 等效应力: {ext['max_mises']:.6e}\n\n"
            f"## AI 工程语境\n"
            f"{context_text or '无'}\n\n"
            f"## AI 点位结果联动\n"
            f"{self._geotech_point_report_text or '无'}\n\n"
            f"## 结论建议\n"
            f"- 若最大位移和应力超过设计阈值，请优先细化局部网格并复核边界条件。\n"
            f"- 建议结合现场参数与规范条文进一步校核安全储备。\n"
        )
        self._last_generated_report = report
        self._report_output.setPlainText(report)
        self._log("Engineering report generated.")

    def _export_report_text(self) -> None:
        if not self._last_generated_report:
            self._generate_engineering_report()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Engineering Report",
            str(Path.cwd() / "rockfem_report.md"),
            "Markdown (*.md);;Text (*.txt)",
        )
        if not path:
            return
        Path(path).write_text(self._last_generated_report, encoding="utf-8")
        self._log(f"Engineering report exported: {path}")

    def _show_startup_dialog(self) -> None:
        dialog = StartupDialog(self._language, self)
        result = dialog.exec()
        if result != QDialog.Accepted:
            self._create_empty_model()
            return
        if dialog.choice.action == "demo":
            self._load_demo_model()
            return
        if dialog.choice.action == "import":
            if not self._import_model_file():
                self._create_empty_model()
            return
        self._create_empty_model()

    def _create_empty_model(self) -> None:
        self._clear_mapping_state()
        self._reset_scene_project()
        self._part_sketch_points.clear()
        self._part_sketch_history.clear()
        self._part_sketch_curve_hint = None
        self._part_sketch_undo_stack.clear()
        self._part_sketch_redo_stack.clear()
        self._material_geometry_region_rules.clear()
        self._selected_face_region_sketch_keys.clear()
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._capture_sketch_from_canvas = False
        self._part_capture_from_canvas.setChecked(False)
        self._mesh_canvas.clear_sketch_plane()
        self._set_model(Model(mesh=Mesh()))
        self._log("Created empty model.")

    def _toggle_log_panel(self) -> None:
        visible = not self._command_history.isVisible()
        self._command_history.setVisible(visible)
        self._on_log_visibility_changed(visible)

    def _clear_logs(self) -> None:
        self._log_text.setPlainText("")
        self._command_history.setPlainText("")

    def _on_log_visibility_changed(self, visible: bool) -> None:
        self._btn_toggle_log.setChecked(visible)
        self._log_hint.setText(self._ui("命令历史面板已显示。", "Command history panel is visible.") if visible else self._ui("命令历史面板已隐藏。", "Command history panel is hidden."))

    def _set_runtime_state(self, state: str, detail: str) -> None:
        token = state.lower().strip()
        self._runtime_state = token
        style = status_badge_style(token)
        label = self._state_text(token)
        self._runtime_badge.setText(label)
        self._runtime_badge.setStyleSheet(style)
        self._toolbar_runtime_badge.setText(label)
        self._toolbar_runtime_badge.setStyleSheet(style)
        if hasattr(self, "_brand_runtime_badge"):
            self._brand_runtime_badge.setText(label)
            self._brand_runtime_badge.setStyleSheet(style)
        self._runtime_detail.setText(detail)
        self.statusBar().showMessage(detail)
        self._mesh_canvas.set_runtime_state(label, results_stale=self._results_stale)

    def _set_results_state_text(self) -> None:
        if self._result is None:
            self._results_state.setText(self._tr("label.no_result", "No solved results yet."))
            self._results_state.setStyleSheet(f"color:{PALETTE.text_secondary};")
            return
        if self._results_stale:
            self._results_state.setText(self._tr("label.results_stale", "Results are stale. Model/backend/mapping changed after solve."))
            self._results_state.setStyleSheet(f"color:{PALETTE.warning}; font-weight: 600;")
            return
        self._results_state.setText(self._tr("label.results_current", "Results are current."))
        self._results_state.setStyleSheet(f"color:{PALETTE.success}; font-weight: 600;")

    def _set_solving_controls(self, solving: bool) -> None:
        enabled = not solving
        for widget in (
            self._btn_toolbar_demo,
            self._btn_toolbar_geotech,
            self._btn_toolbar_import,
            self._btn_toolbar_solve,
            self._backend_combo,
            self._btn_apply_mapping,
            self._btn_export_nodes,
            self._btn_export_elements,
            self._results_scope_type_combo,
            self._results_scope_target_combo,
            self._btn_results_scope_clear,
            self._btn_toolbar_export,
            self._language_combo,
            self._btn_quick_undo,
            self._btn_quick_redo,
            self._btn_quick_zoom_in,
            self._btn_quick_zoom_out,
            self._btn_quick_reset_view,
            self._btn_quick_cancel,
            self._btn_part_tool_select,
            self._btn_part_tool_add_point,
            self._btn_part_tool_measure_len,
            self._btn_part_tool_measure_ang,
            self._btn_part_tool_rect,
            self._btn_part_tool_circle,
            self._btn_part_tool_ellipse,
            self._btn_part_tool_close,
            self._btn_part_tool_clear_selected,
            self._btn_part_tool_clear_all,
            self._part_shape_w,
            self._part_shape_h,
            self._part_shape_lock_ratio,
            self._part_quick_mode,
            self._part_quick_value,
            self._btn_part_quick_apply,
            self._btn_geotech_quick,
            self._btn_geotech_text,
            self._btn_geotech_csv,
            self._btn_geotech_rerun,
            self._btn_geotech_open_output,
            self._btn_editor_load,
            self._btn_editor_validate,
            self._btn_editor_apply,
            self._btn_editor_add_material,
            self._btn_editor_del_material,
            self._btn_editor_add_node,
            self._btn_editor_del_node,
            self._btn_editor_add_element,
            self._btn_editor_del_element,
            self._editor_canvas_edit_enable,
            self._editor_canvas_tool_combo,
            self._editor_canvas_material_spin,
            self._geotech_mesh_size_spin,
            self._geotech_top_load_spin,
            self._geotech_fix_bottom_ux,
            self._geotech_fix_bottom_uy,
            self._geotech_fix_lateral_ux,
            self._geotech_export_artifacts,
            self._btn_rock_ai_image,
            self._btn_rock_ai_context,
            self._btn_rock_create_2d,
            self._btn_model_measure,
            self._btn_model_reset_view,
            self._btn_part_add_point,
            self._part_capture_from_canvas,
            self._part_snap_to_grid,
            self._btn_part_measure,
            self._btn_part_measure_fill_length,
            self._part_length_target,
            self._btn_part_apply_length,
            self._part_angle_target,
            self._btn_part_apply_angle,
            self._btn_part_constraint_horizontal,
            self._btn_part_constraint_vertical,
            self._btn_part_constraint_collinear,
            self._btn_part_constraint_parallel,
            self._btn_part_constraint_perpendicular,
            self._btn_part_clear_selected,
            self._btn_part_clear,
            self._btn_part_close,
            self._btn_part_build,
            self._material_template_combo,
            self._btn_material_apply_template,
            self._btn_material_create,
            self._btn_material_rename,
            self._btn_material_duplicate,
            self._btn_material_delete,
            self._btn_material_pick_face,
            self._btn_material_assign,
            self._material_assign_combo,
            self._material_toolbar_combo,
            self._material_reassign_target_combo,
            self._btn_material_reassign_all,
            self._btn_material_tool_create,
            self._material_toolbar_template_combo,
            self._btn_material_tool_apply_template,
            self._btn_material_tool_rename,
            self._btn_material_tool_duplicate,
            self._btn_material_tool_pick_face,
            self._btn_material_tool_assign,
            self._btn_material_tool_delete,
            self._material_action_template_combo,
            self._btn_material_action_apply_template,
            self._btn_material_action_create,
            self._btn_material_action_rename,
            self._btn_material_action_duplicate,
            self._btn_material_action_delete,
            self._btn_material_action_pick_face,
            self._btn_material_action_assign,
            self._btn_component_pick_face,
            self._btn_component_create_from_selected,
            self._btn_component_rename,
            self._btn_component_toggle_visibility,
            self._btn_component_isolate,
            self._btn_component_show_all,
            self._btn_apply_assembly_offset,
            self._load_bc_target_mode_combo,
            self._load_bc_target_set_combo,
            self._btn_load_bc_capture_set,
            self._btn_load_bc_capture_region_set,
            self._btn_load_bc_capture_component_set,
            self._btn_load_bc_pick_geo_point,
            self._btn_load_bc_pick_geo_edge,
            self._btn_load_bc_pick_geo_point_set,
            self._btn_load_bc_split_edge_point,
            self._btn_load_bc_clear_point_set,
            self._btn_load_bc_finish_pick,
            self._btn_load_bc_cleanup_points,
            self._btn_pick_load_point,
            self._btn_add_point_load,
            self._btn_pick_dist_edge,
            self._btn_add_dist_load,
            self._btn_pick_bc_point,
            self._btn_pick_bc_edge,
            self._btn_bc_fixed,
            self._btn_bc_fix_x,
            self._btn_bc_fix_y,
            self._btn_apply_bc,
            self._btn_generate_mesh,
            self._btn_import_mesh,
            self._btn_mesh_pick_region,
            self._btn_mesh_apply_region_seed,
            self._btn_mesh_pick_edge,
            self._btn_mesh_apply_edge_seed,
            self._btn_mesh_generate_selection,
            self._btn_mesh_remesh_selection,
            self._btn_mesh_delete_selection,
            self._btn_mesh_quality_check,
            self._btn_mesh_toggle_bad,
            self._btn_submit_job,
            self._btn_job_export_canvas,
            self._btn_generate_report,
            self._btn_export_report,
        ):
            widget.setEnabled(enabled)

    def _set_model(self, model: Model) -> None:
        self._model = model
        self._result = None
        self._results_stale = False
        self._pending_pick_context = None
        self._selected_node_ids.clear()
        self._selected_element_ids.clear()
        self._selected_face_region_element_ids.clear()
        self._selected_face_region_sketch_keys.clear()
        self._selected_geometry_point_ids.clear()
        self._selected_geometry_edge_ids.clear()
        self._mesh_canvas.set_model(model)
        self._sync_material_sketch_faces_to_canvas()
        self._mesh_canvas.set_highlighted_sketch_faces([])
        if model.mesh.nodes:
            self._mesh_canvas.clear_sketch_plane()
        self._mesh_canvas.set_result(None)
        if hasattr(self, "_material_face_hint"):
            self._material_face_hint.setText(self._tr("rock.material.pick_face.hint", "未选择区域"))
        if hasattr(self, "_component_face_hint"):
            self._component_face_hint.setText(self._tr("rock.assembly.component.pick.hint", "未选择区域"))
        self._sync_pickable_sketch_faces_to_canvas()
        self._clear_result_views()
        self._sync_editor_tables_from_model()
        self._update_model_tree()
        self._rebuild_scene_mesh_state()
        self._last_mesh_quality_report = evaluate_t3_mesh_quality(model.mesh) if model.mesh.elements else None
        self._scene_mesh_state.quality_report = self._last_mesh_quality_report
        unresolved_definitions = self._resolve_scene_load_bc_definitions_to_model()
        if unresolved_definitions and (self._scene_project.load_definitions or self._scene_project.boundary_definitions):
            self._log(
                self._ui(
                    "部分几何载荷/约束尚未解析，生成保点网格后会再次解析: ",
                    "Some geometry Load/BC definitions are unresolved and will be retried after geometry-preserving meshing: ",
                )
                + " | ".join(unresolved_definitions[:3])
            )
        self._sync_editor_tables_from_model()
        self._update_model_tree()
        self._refresh_component_manager()
        self._sync_component_sketch_faces_to_canvas()
        self._apply_component_visibility_filter_to_canvas()
        self._refresh_load_bc_target_set_combo()
        self._refresh_load_bc_managers()
        self._update_status_labels()
        self._set_results_state_text()
        self._refresh_readiness_state()
        self._update_visualization_metrics()
        self._update_mesh_selection_hint()
        self._refresh_mesh_quality_summary()

    def _clear_result_views(self) -> None:
        self._summary_text.setPlainText(self._tr("label.no_result", "No solved results yet."))
        self._geotech_point_report_text = ""
        self._geotech_last_output_dir = None
        self._geotech_point_summary.setText(self._tr("geotech.points.empty", "No geotechnical point report yet."))
        self._node_rows = []
        self._element_rows = []
        self._node_table.clearSelection()
        self._element_table.clearSelection()
        self._mesh_canvas.set_highlighted_nodes(set())
        self._mesh_canvas.set_highlighted_elements(set())
        self._node_selection_label.setText(self._tr("node.selection.empty", "Select a node result row to highlight mesh nodes."))
        self._element_selection_label.setText(self._tr("element.selection.empty", "Select an element result row to highlight mesh elements."))
        self._refresh_node_table()
        self._refresh_element_table()
        self._update_visualization_metrics()

    def _clear_mapping_state(self) -> None:
        self._last_gmsh_import = None
        self._mapping_entries = []
        self._mapping_applied = False
        self._update_mapping_preview([])

    def _load_demo_model(self) -> None:
        model = build_stage1_demo_model()
        self._clear_mapping_state()
        self._reset_scene_project()
        self._part_sketch_points.clear()
        self._part_sketch_history.clear()
        self._part_sketch_curve_hint = None
        self._part_sketch_undo_stack.clear()
        self._part_sketch_redo_stack.clear()
        self._material_geometry_region_rules.clear()
        self._selected_face_region_sketch_keys.clear()
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._set_model(model)
        self._seed_scene_from_imported_model()
        self._log("Loaded stage1 demo model.")

    def _default_geotech_template_text(self) -> str:
        return (
            "layer Fill: top=[(0,0),(30,0)] bottom=[(0,-4),(30,-6)] E=2.0e7 nu=0.30\n"
            "layer ClayRock: top=[(0,-4),(30,-6)] bottom=[(0,-12),(30,-12)] E=3.7e7 nu=0.29\n"
            "layer WeatheredRock: top=[(0,-12),(30,-12)] bottom=[(0,-20),(30,-20)] E=5.0e7 nu=0.27\n"
            "Q=35\n"
            "mesh_size=1.2\n"
            "point A=(2.0,-3.8)\n"
            "point B=(20.0,-5.5)\n"
        )

    def _default_geotech_points_from_layers(self, layers) -> list[PointOfInterest]:
        all_points = [
            point
            for layer in layers
            for point in (layer.top_boundary.points + layer.bottom_boundary.points)
        ]
        if not all_points:
            return []
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        width = max_x - min_x
        height = max_y - min_y
        return [
            PointOfInterest(name="A", x=min_x + 0.12 * width, y=max_y - 0.22 * height),
            PointOfInterest(name="B", x=min_x + 0.70 * width, y=max_y - 0.28 * height),
        ]

    def _apply_geotech_ui_overrides(self, template: GeotechTemplateInput) -> GeotechTemplateInput:
        """Apply UI mesh/boundary overrides to a geotechnical template."""
        template.mesh_target_size = float(self._geotech_mesh_size_spin.value())
        template.top_line_load = float(self._geotech_top_load_spin.value())
        template.fix_bottom_ux = bool(self._geotech_fix_bottom_ux.isChecked())
        template.fix_bottom_uy = bool(self._geotech_fix_bottom_uy.isChecked())
        template.constrain_lateral_ux = bool(self._geotech_fix_lateral_ux.isChecked())
        return template

    def _resolve_geotech_output_dir(self) -> Path | None:
        if not self._geotech_export_artifacts.isChecked():
            return None
        return Path.cwd() / "output" / "geotech_ui"

    def _rerun_last_geotech_template(self) -> None:
        if self._geotech_last_template is None:
            self._show_error(
                self._tr("geotech.rerun.no_template.title", "No geotechnical template"),
                self._tr("geotech.rerun.no_template.msg", "Run a geotechnical template once before rerun."),
            )
            return
        template = copy.deepcopy(self._geotech_last_template)
        self._run_geotech_template(template)

    def _open_geotech_output_folder(self) -> None:
        if not self._geotech_last_output_dir:
            self._show_error(
                self._tr("geotech.output.none.title", "No output folder"),
                self._tr("geotech.output.none.msg", "No exported geotechnical output is available yet."),
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._geotech_last_output_dir))

    def _append_editor_row(self, table: QTableWidget, values: list[str]) -> None:
        row = table.rowCount()
        table.setRowCount(row + 1)
        for col, value in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(value))

    def _delete_selected_editor_rows(self, table: QTableWidget) -> None:
        rows = sorted({item.row() for item in table.selectedItems()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    def _table_cell_text(self, table: QTableWidget, row: int, col: int) -> str:
        item = table.item(row, col)
        return item.text().strip() if item is not None else ""

    def _sync_editor_tables_from_model(self) -> None:
        model = self._model
        self._editor_material_table.setRowCount(0)
        self._editor_node_table.setRowCount(0)
        self._editor_element_table.setRowCount(0)

        if model is None:
            return

        material_ids: list[int] = []
        for material in sorted(model.materials, key=lambda item: item.id):
            material_ids.append(material.id)
            self._append_editor_row(
                self._editor_material_table,
                [
                    str(material.id),
                    f"{material.young_modulus:.6g}",
                    f"{material.poisson_ratio:.6g}",
                    "true" if material.plane_stress else "false",
                ],
            )

        if material_ids:
            max_material_id = max(material_ids)
            self._editor_canvas_material_spin.setMaximum(max(1_000_000, max_material_id + 1000))
            if self._editor_canvas_material_spin.value() not in material_ids:
                self._editor_canvas_material_spin.setValue(max_material_id)

        for node in sorted(model.mesh.nodes, key=lambda item: item.id):
            self._append_editor_row(
                self._editor_node_table,
                [str(node.id), f"{node.x:.6g}", f"{node.y:.6g}"],
            )

        for element in sorted(model.mesh.elements, key=lambda item: item.id):
            if len(element.connectivity) != 3:
                continue
            self._append_editor_row(
                self._editor_element_table,
                [
                    str(element.id),
                    str(element.connectivity[0]),
                    str(element.connectivity[1]),
                    str(element.connectivity[2]),
                    str(element.material_id),
                ],
            )

    def _parse_editor_bool(self, text: str) -> bool:
        token = text.strip().lower()
        return token in {"1", "true", "yes", "y", "on"}

    def _build_model_from_editor_tables(self) -> Model:
        materials: list[Material] = []
        nodes: list[Node] = []
        elements: list[Element] = []

        for row in range(self._editor_material_table.rowCount()):
            values = [self._table_cell_text(self._editor_material_table, row, col) for col in range(4)]
            if not any(values):
                continue
            try:
                materials.append(
                    Material(
                        id=int(values[0]),
                        young_modulus=float(values[1]),
                        poisson_ratio=float(values[2]),
                        plane_stress=self._parse_editor_bool(values[3]),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid material row {row + 1}: {values}") from exc

        for row in range(self._editor_node_table.rowCount()):
            values = [self._table_cell_text(self._editor_node_table, row, col) for col in range(3)]
            if not any(values):
                continue
            try:
                nodes.append(Node(id=int(values[0]), x=float(values[1]), y=float(values[2])))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid node row {row + 1}: {values}") from exc

        for row in range(self._editor_element_table.rowCount()):
            values = [self._table_cell_text(self._editor_element_table, row, col) for col in range(5)]
            if not any(values):
                continue
            try:
                elements.append(
                    Element(
                        id=int(values[0]),
                        type="T3",
                        connectivity=[int(values[1]), int(values[2]), int(values[3])],
                        material_id=int(values[4]),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid element row {row + 1}: {values}") from exc

        draft = Model(mesh=Mesh(nodes=nodes, elements=elements), materials=materials)

        # Preserve valid BC/load entries from current model to keep workflow continuity.
        existing_model = self._model
        if existing_model is not None:
            node_ids = {node.id for node in nodes}
            draft.boundary_conditions = [
                BoundaryCondition(node_id=bc.node_id, dof=bc.dof, value=bc.value)
                for bc in existing_model.boundary_conditions
                if bc.node_id in node_ids
            ]
            draft.loads = [
                Load(node_id=load.node_id, dof=load.dof, value=load.value)
                for load in existing_model.loads
                if load.node_id in node_ids
            ]

        return draft

    def _validate_editor_draft(self) -> None:
        try:
            draft = self._build_model_from_editor_tables()
            validate_model(draft)
        except Exception as exc:  # pragma: no cover
            self._show_error(
                self._tr("editor.validate.error.title", "Editor draft invalid"),
                str(exc),
            )
            self._set_runtime_state("warning", f"Editor draft invalid: {exc}")
            return
        self._set_runtime_state("ready", self._tr("editor.validate.ok", "Editor draft is valid for solving."))
        self._log(self._tr("editor.validate.ok", "Editor draft is valid for solving."))

    def _apply_editor_draft_to_model(self) -> None:
        try:
            draft = self._build_model_from_editor_tables()
            validate_model(draft)
        except Exception as exc:  # pragma: no cover
            self._show_error(
                self._tr("editor.apply.error.title", "Failed to apply editor draft"),
                str(exc),
            )
            self._set_runtime_state("warning", f"Failed to apply editor draft: {exc}")
            return
        self._clear_mapping_state()
        self._reset_scene_project()
        self._material_geometry_region_rules.clear()
        self._selected_face_region_sketch_keys.clear()
        self._set_model(draft)
        self._seed_scene_from_imported_model()
        self._log(self._tr("editor.apply.ok", "Editor draft applied to current model."))

    def _selected_canvas_tool(self) -> str:
        value = self._editor_canvas_tool_combo.currentData()
        if isinstance(value, str):
            return value
        return "select"

    def _on_canvas_edit_enabled_changed(self, enabled: bool) -> None:
        self._mesh_canvas.set_edit_enabled(bool(enabled))
        self._mesh_canvas.set_edit_tool(self._selected_canvas_tool())
        if not enabled:
            self._pending_pick_context = None

    def _on_canvas_edit_tool_changed(self, _: int) -> None:
        self._mesh_canvas.set_edit_tool(self._selected_canvas_tool())
        current_tool = self._selected_canvas_tool()
        if (
            current_tool == "pick_geometry"
            and self._pending_pick_context in {
                "geometry_point_target",
                "geometry_edge_target",
                "geometry_point_set_multi",
                "geometry_split_edge_point",
                "mesh_edge_seed",
            }
        ):
            return
        if current_tool != "select" and self._pending_pick_context in {
            "material_face",
            "material_face_sketch",
            "component_face",
            "component_face_sketch",
            "mesh_region",
            "mesh_region_sketch",
            "mesh_edge_seed",
            "load_point",
            "bc_point",
            "geometry_point_target",
            "geometry_edge_target",
            "geometry_point_set_multi",
            "geometry_split_edge_point",
        }:
            if self._pending_pick_context in {"material_face", "material_face_sketch"} and hasattr(self, "_material_face_hint"):
                self._material_face_hint.setText(self._tr("rock.material.pick_face.hint", "未选择区域"))
            if self._pending_pick_context in {"component_face", "component_face_sketch"} and hasattr(self, "_component_face_hint"):
                self._component_face_hint.setText(self._tr("rock.assembly.component.pick.hint", "未选择区域"))
            if self._pending_pick_context in {"mesh_region", "mesh_region_sketch", "mesh_edge_seed"}:
                self._update_mesh_selection_hint()
            self._pending_pick_context = None
            self._sync_pickable_sketch_faces_to_canvas()
            self._sync_load_bc_geometry_overlay()

    def _next_node_id(self) -> int:
        if self._model is None or not self._model.mesh.nodes:
            return 1
        return max(node.id for node in self._model.mesh.nodes) + 1

    def _next_element_id(self) -> int:
        if self._model is None or not self._model.mesh.elements:
            return 1
        return max(element.id for element in self._model.mesh.elements) + 1

    def _ensure_material_for_new_elements(self, material_id: int) -> None:
        if self._model is None:
            return
        if any(material.id == material_id for material in self._model.materials):
            return
        if self._model.materials:
            base = self._model.materials[0]
            material = Material(
                id=material_id,
                young_modulus=base.young_modulus,
                poisson_ratio=base.poisson_ratio,
                plane_stress=base.plane_stress,
            )
        else:
            material = Material(
                id=material_id,
                young_modulus=2.0e7,
                poisson_ratio=0.30,
                plane_stress=False,
            )
        self._model.materials.append(material)
        self._model.materials.sort(key=lambda item: item.id)
        if material_id not in self._material_id_to_name:
            name = f"MAT-{material_id}"
            self._material_id_to_name[material_id] = name
            self._material_name_to_id[name] = material_id
        self._update_model_tree()
        self._update_status_labels()

    def _get_node_by_id(self, node_id: int) -> Node | None:
        if self._model is None:
            return None
        for node in self._model.mesh.nodes:
            if node.id == node_id:
                return node
        return None

    def _signed_double_area_from_ids(self, n1: int, n2: int, n3: int) -> float:
        node1 = self._get_node_by_id(n1)
        node2 = self._get_node_by_id(n2)
        node3 = self._get_node_by_id(n3)
        if node1 is None or node2 is None or node3 is None:
            raise ValueError("Element references unknown node id.")
        return (node2.x - node1.x) * (node3.y - node1.y) - (node3.x - node1.x) * (node2.y - node1.y)

    def _on_canvas_node_created(self, x: float, y: float) -> None:
        if self._capture_sketch_from_canvas or self._mesh_canvas.is_sketch_plane_active():
            sx, sy = self._snap_sketch_point(float(x), float(y))
            self._push_part_undo_snapshot()
            self._clear_part_curve_hint()
            self._selected_part_segment = None
            self._selected_part_angle = None
            self._part_sketch_points.append((sx, sy))
            self._part_point_x.setValue(sx)
            self._part_point_y.setValue(sy)
            self._refresh_part_sketch_table()
            self._log(self._ui(f"草图点已添加: ({sx:.4f}, {sy:.4f})", f"Sketch point added: ({sx:.4f}, {sy:.4f})"))
            return

        if self._model is None:
            self._create_empty_model()
        if self._model is None:
            return
        node_id = self._next_node_id()
        self._model.mesh.nodes.append(Node(id=node_id, x=float(x), y=float(y)))
        self._model.mesh.nodes.sort(key=lambda item: item.id)
        self._set_model(self._model)
        self._mesh_canvas.set_highlighted_nodes({node_id})
        self._log(f"Canvas edit: added node {node_id} at ({x:.4f}, {y:.4f}).")

    def _on_canvas_node_moved(self, node_id: int, x: float, y: float) -> None:
        if self._model is None:
            return
        node = self._get_node_by_id(node_id)
        if node is None:
            return
        node.x = float(x)
        node.y = float(y)
        self._set_model(self._model)
        self._mesh_canvas.set_highlighted_nodes({node_id})
        self._mark_results_stale("Mesh node moved on canvas")
        self._log(f"Canvas edit: moved node {node_id} to ({x:.4f}, {y:.4f}).")

    def _on_canvas_element_created(self, node_ids: list[int]) -> None:
        if self._model is None:
            return
        unique_ids = []
        for node_id in node_ids:
            if node_id not in unique_ids:
                unique_ids.append(int(node_id))
        if len(unique_ids) != 3:
            self._show_error(
                self._tr("editor.canvas.element.error.title", "Failed to create element"),
                self._tr("editor.canvas.element.error.nodes", "Element creation requires 3 unique node ids."),
            )
            return

        try:
            signed_two_area = self._signed_double_area_from_ids(unique_ids[0], unique_ids[1], unique_ids[2])
        except Exception as exc:
            self._show_error(self._tr("editor.canvas.element.error.title", "Failed to create element"), str(exc))
            return

        if abs(signed_two_area) <= 1e-14:
            self._show_error(
                self._tr("editor.canvas.element.error.title", "Failed to create element"),
                self._tr("editor.canvas.element.error.degenerate", "Selected nodes form a degenerate triangle."),
            )
            return
        if signed_two_area < 0.0:
            unique_ids = [unique_ids[0], unique_ids[2], unique_ids[1]]

        material_id = int(self._editor_canvas_material_spin.value())
        self._ensure_material_for_new_elements(material_id)
        element_id = self._next_element_id()
        self._model.mesh.elements.append(
            Element(
                id=element_id,
                type="T3",
                connectivity=unique_ids,
                material_id=material_id,
            )
        )
        self._model.mesh.elements.sort(key=lambda item: item.id)
        self._set_model(self._model)
        self._mesh_canvas.set_highlighted_elements({element_id})
        self._mark_results_stale("Mesh element added on canvas")
        self._log(
            f"Canvas edit: added element {element_id} with nodes {unique_ids} and material {material_id}."
        )

    def _apply_geotech_workflow_result(self, run_result, *, backend: str) -> None:
        self._clear_mapping_state()
        self._reset_scene_project()
        self._part_sketch_points.clear()
        self._part_sketch_history.clear()
        self._part_sketch_curve_hint = None
        self._part_sketch_undo_stack.clear()
        self._part_sketch_redo_stack.clear()
        self._material_geometry_region_rules.clear()
        self._selected_face_region_sketch_keys.clear()
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._set_model(run_result.model)
        self._seed_scene_from_imported_model()
        self._result = run_result.solve_result
        self._results_stale = False
        self._mesh_canvas.set_result(run_result.solve_result)
        self._summary_text.setPlainText(self._format_summary(run_result.solve_result, backend))
        self._populate_result_rows(run_result.solve_result)
        self._update_model_tree()
        self._update_status_labels()
        self._set_results_state_text()
        self._update_visualization_metrics()

        if run_result.point_results:
            lines = [
                f"{item.name}: ux={item.ux:.4e}, uy={item.uy:.4e}, node={item.node_id}"
                for item in run_result.point_results
            ]
            self._geotech_point_report_text = " | ".join(lines)
        else:
            self._geotech_point_report_text = self._tr(
                "geotech.points.none",
                "No A/B points defined in geotechnical template.",
            )

        if run_result.output_files is not None:
            self._geotech_last_output_dir = str(run_result.output_files.base_dir)
            self._log(
                f"Geotech outputs saved: {self._geotech_last_output_dir}"
            )
        else:
            self._geotech_last_output_dir = None

        self._geotech_point_summary.setText(self._geotech_point_report_text)
        self._set_runtime_state(
            "solved",
            self._tr("geotech.solve.done", "Geotechnical template solved successfully."),
        )
        self._log(self._tr("geotech.solve.done", "Geotechnical template solved successfully."))

    def _run_geotech_template(self, template: GeotechTemplateInput) -> None:
        backend = self._backend_combo.currentText()
        runtime_template = self._apply_geotech_ui_overrides(copy.deepcopy(template))
        runtime_template.backend = backend
        output_dir = self._resolve_geotech_output_dir()
        self._geotech_last_template = copy.deepcopy(runtime_template)

        self._set_solving_controls(True)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._set_runtime_state("solving", self._tr("geotech.solve.running", "Running geotechnical template workflow..."))
        QApplication.processEvents()
        try:
            run_result = run_geotech_template_workflow(runtime_template, output_dir=output_dir)
        except Exception as exc:  # pragma: no cover
            self._show_error(
                self._tr("geotech.solve.error.title", "Geotechnical workflow failed"),
                str(exc),
            )
            self._set_runtime_state("error", f"Geotechnical workflow failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._set_solving_controls(False)

        self._apply_geotech_workflow_result(run_result, backend=backend)

    def _run_geotech_quick_template(self) -> None:
        template = build_template_input_from_text_description(
            self._default_geotech_template_text(),
            backend=self._backend_combo.currentText(),
        )
        self._run_geotech_template(template)

    def _run_geotech_from_text_dialog(self) -> None:
        text, accepted = QInputDialog.getMultiLineText(
            self,
            self._tr("geotech.text.title", "Geotechnical Text Template"),
            self._tr(
                "geotech.text.prompt",
                "Edit layered section text and run geotechnical template workflow:",
            ),
            self._default_geotech_template_text(),
        )
        if not accepted or not text.strip():
            return
        try:
            template = build_template_input_from_text_description(
                text,
                backend=self._backend_combo.currentText(),
            )
        except Exception as exc:
            self._show_error(self._tr("geotech.text.parse_error", "Failed to parse geotechnical text"), str(exc))
            return
        self._run_geotech_template(template)

    def _run_geotech_from_csv_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("geotech.csv.title", "Import Geotechnical Layer CSV"),
            str(Path.cwd()),
            "CSV (*.csv)",
        )
        if not file_path:
            return
        try:
            layers = load_geotech_layers_from_csv(file_path)
        except Exception as exc:
            self._show_error(self._tr("geotech.csv.parse_error", "Failed to parse geotechnical CSV"), str(exc))
            return

        poi = self._default_geotech_points_from_layers(layers)
        template = GeotechTemplateInput(
            layers=layers,
            mesh_target_size=float(self._geotech_mesh_size_spin.value()),
            top_line_load=float(self._geotech_top_load_spin.value()),
            points_of_interest=poi,
            backend=self._backend_combo.currentText(),
        )
        self._run_geotech_template(template)

    def _import_model_file(self) -> bool:
        file_path, _ = QFileDialog.getOpenFileName(self, self._tr("startup.import_model", "Import Model File"), str(Path.cwd()), "Mesh (*.msh *.json);;Gmsh (*.msh);;JSON (*.json)")
        if not file_path:
            return False
        suffix = Path(file_path).suffix.lower()
        if suffix == ".json":
            return self._import_json_mesh_from_path(file_path)
        if suffix == ".msh":
            return self._import_gmsh_mesh_from_path(file_path)
        self._show_error(self._tr("error.import", "Import failed"), f"Unsupported file format: {suffix}")
        return False

    def _import_json_mesh(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, self._tr("toolbar.import_json", "Import T3 JSON Mesh"), str(Path.cwd()), "JSON (*.json)")
        if file_path:
            self._import_json_mesh_from_path(file_path)

    def _import_json_mesh_from_path(self, file_path: str) -> bool:
        try:
            mesh = load_t3_mesh_from_json(file_path)
            model = model_from_t3_mesh(mesh)
            apply_left_clamp_right_nodal_force(model)
            validate_model(model)
        except Exception as exc:  # pragma: no cover
            self._show_error(self._tr("error.import", "Import failed"), str(exc))
            self._set_runtime_state("error", f"JSON import failed: {exc}")
            return False
        self._clear_mapping_state()
        self._reset_scene_project()
        self._part_sketch_points.clear()
        self._part_sketch_history.clear()
        self._part_sketch_curve_hint = None
        self._part_sketch_undo_stack.clear()
        self._part_sketch_redo_stack.clear()
        self._material_geometry_region_rules.clear()
        self._selected_face_region_sketch_keys.clear()
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._set_model(model)
        self._seed_scene_from_imported_model()
        self._log(f"Imported JSON mesh: {file_path}")
        return True

    def _import_gmsh_mesh(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, self._tr("toolbar.import_gmsh", "Import Gmsh .msh"), str(Path.cwd()), "Gmsh (*.msh)")
        if file_path:
            self._import_gmsh_mesh_from_path(file_path)

    def _import_gmsh_mesh_from_path(self, file_path: str) -> bool:
        try:
            gmsh_import = load_t3_mesh_from_gmsh_msh_with_physical_groups(file_path)
            model = model_from_t3_mesh(gmsh_import.mesh, preserve_element_material_ids=True)
            mapping_entries = build_physical_group_mapping_preview(gmsh_import)
            self._mapping_entries = mapping_entries
            self._last_gmsh_import = gmsh_import
            self._mapping_applied = False
            self._update_mapping_preview(mapping_entries)
            if self._confirm_mapping_apply(mapping_entries):
                report = self._apply_mapping_templates(model, gmsh_import)
                self._mapping_entries = report.mapping_entries
                self._mapping_applied = True
                self._update_mapping_preview(report.mapping_entries)
            validate_model(model)
        except Exception as exc:  # pragma: no cover
            self._show_error(self._tr("error.import", "Import failed"), str(exc))
            self._set_runtime_state("error", f"Gmsh import failed: {exc}")
            return False
        self._reset_scene_project()
        self._part_sketch_points.clear()
        self._part_sketch_history.clear()
        self._part_sketch_curve_hint = None
        self._part_sketch_undo_stack.clear()
        self._part_sketch_redo_stack.clear()
        self._material_geometry_region_rules.clear()
        self._selected_face_region_sketch_keys.clear()
        self._reset_part_sketch_selection()
        self._refresh_part_sketch_table()
        self._set_model(model)
        self._seed_scene_from_imported_model(gmsh_import=gmsh_import)
        self._log(f"Imported Gmsh mesh: {file_path}")
        return True

    def _confirm_mapping_apply(self, entries: list[PhysicalGroupMappingEntry]) -> bool:
        counts = summarize_mapping_entries(entries)
        msg = f"Detected groups: {counts['total_group_count']}\nMapped: {counts['mapped_group_count']}\nWarning: {counts['warning_group_count']}\nUnrecognized: {counts['unrecognized_group_count']}\n\nApply recognized templates now?"
        ans = QMessageBox.question(self, "Physical Group Mapping", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return ans == QMessageBox.Yes

    def _apply_mapping_templates(self, model: Model, gmsh_import: GmshT3ImportResult) -> TemplateApplicationReport:
        report = apply_default_templates_from_physical_groups(model, gmsh_import)
        if report.boundary_condition_count == 0 and report.load_count == 0:
            apply_left_clamp_right_nodal_force(model)
        return report

    def _apply_mapping_from_last_import(self) -> None:
        if self._model is None or self._last_gmsh_import is None:
            return
        try:
            report = self._apply_mapping_templates(self._model, self._last_gmsh_import)
            validate_model(self._model)
        except Exception as exc:  # pragma: no cover
            self._show_error("Apply mapping failed", str(exc))
            return
        self._mapping_entries = report.mapping_entries
        self._mapping_applied = True
        self._update_mapping_preview(report.mapping_entries)
        self._rebuild_scene_mesh_state()
        self._update_model_tree()
        self._update_status_labels()
        self._mark_results_stale("Mapping templates updated")

    def _solve(self) -> None:
        if self._model is None:
            self._show_error("Solve failed", "No model available")
            return
        self._rebuild_scene_mesh_state()
        unresolved_definitions = self._resolve_scene_load_bc_definitions_to_model()
        if unresolved_definitions:
            self._show_error(
                self._ui("求解失败", "Solve failed"),
                self._ui(
                    "存在未解析到网格节点的几何载荷/约束，请先生成保点网格或重新选择目标：\n"
                    + "\n".join(unresolved_definitions[:6]),
                    "Some geometry Load/BC definitions are unresolved. Generate a geometry-preserving mesh or re-pick targets first:\n"
                    + "\n".join(unresolved_definitions[:6]),
                ),
            )
            return

        backend = self._backend_combo.currentText()
        self._set_solving_controls(True)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._set_runtime_state("solving", f"Solving model with backend='{backend}'...")
        QApplication.processEvents()
        try:
            validate_model(self._model)
            result = solve_linear_static(self._model, backend=backend)
        except Exception as exc:  # pragma: no cover
            self._show_error("Solve failed", str(exc))
            self._set_runtime_state("error", f"Solve failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._set_solving_controls(False)

        self._result = result
        self._results_stale = False
        self._mesh_canvas.set_result(result)
        self._summary_text.setPlainText(self._format_summary(result, backend))
        self._populate_result_rows(result)
        self._update_model_tree()
        self._update_status_labels()
        self._set_results_state_text()
        self._update_visualization_metrics()
        self._set_runtime_state("solved", f"Solved successfully. Max |u| = {result.summary.max_displacement:.3e}")

    def _refresh_readiness_state(self) -> None:
        warnings: list[str] = []
        if self._model is not None:
            try:
                validate_model(self._model)
            except Exception as exc:
                warnings.append(str(exc))
            warnings.extend(collect_pre_solve_warnings(self._model))
        if warnings:
            self._set_runtime_state("warning", warnings[0])
            self._pre_solve_warning.setText("Warnings: " + " | ".join(warnings[:3]))
        else:
            self._set_runtime_state("ready", "Model is ready to solve.")
            self._pre_solve_warning.setText("No blocking pre-solve warnings.")

    def _mark_results_stale(self, reason: str) -> None:
        if self._result is None or self._results_stale:
            return
        self._results_stale = True
        self._set_results_state_text()
        self._set_runtime_state("warning", f"Results stale: {reason}")

    def _populate_result_rows(self, result: StaticSolveResult) -> None:
        self._node_rows = build_node_rows(result)
        self._element_rows = build_element_rows(result)
        self._refresh_results_scope_targets()

    def _refresh_node_table(self) -> None:
        allowed_node_ids = self._results_scope_node_ids()
        base_indices = [
            idx
            for idx, row in enumerate(self._node_rows)
            if allowed_node_ids is None or int(row[0]) in allowed_node_ids
        ]
        base_rows = [self._node_rows[idx] for idx in base_indices]
        visible_rel = filter_row_indices(base_rows, self._node_filter_edit.text())
        visible = [base_indices[idx] for idx in visible_rel]
        self._node_visible_indices = visible
        self._node_table.setSortingEnabled(False)
        self._node_table.setRowCount(len(visible))
        for row, index in enumerate(visible):
            node_id, ux, uy = self._node_rows[index]
            self._node_table.setItem(row, 0, QTableWidgetItem(str(node_id)))
            self._node_table.setItem(row, 1, QTableWidgetItem(f"{ux:.6e}"))
            self._node_table.setItem(row, 2, QTableWidgetItem(f"{uy:.6e}"))
        self._node_table.setSortingEnabled(True)

    def _refresh_element_table(self) -> None:
        allowed_element_ids = self._results_scope_element_ids()
        base_indices = [
            idx
            for idx, row in enumerate(self._element_rows)
            if allowed_element_ids is None or int(row[0]) in allowed_element_ids
        ]
        base_rows = [self._element_rows[idx] for idx in base_indices]
        visible_rel = filter_row_indices(base_rows, self._element_filter_edit.text())
        visible = [base_indices[idx] for idx in visible_rel]
        self._element_visible_indices = visible
        self._element_table.setSortingEnabled(False)
        self._element_table.setRowCount(len(visible))
        for row, index in enumerate(visible):
            values = self._element_rows[index]
            for col, value in enumerate(values):
                if col == 0:
                    text = str(value)
                else:
                    text = f"{float(value):.6e}"
                self._element_table.setItem(row, col, QTableWidgetItem(text))
        self._element_table.setSortingEnabled(True)

    def _export_current_results(self) -> None:
        if self._results_tabs.currentIndex() == 1:
            self._export_node_table()
        elif self._results_tabs.currentIndex() == 2:
            self._export_element_table()

    def _export_node_table(self) -> None:
        if not self._node_visible_indices:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Node Table", str(Path.cwd() / "node_displacements.csv"), "CSV (*.csv)")
        if not path:
            return
        rows = [self._node_rows[i] for i in self._node_visible_indices]
        export_rows_to_csv(path, ["node_id", "ux", "uy"], rows)

    def _export_element_table(self) -> None:
        if not self._element_visible_indices:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Element Table", str(Path.cwd() / "element_results.csv"), "CSV (*.csv)")
        if not path:
            return
        rows = [self._element_rows[i] for i in self._element_visible_indices]
        export_rows_to_csv(path, ["element_id", "ex", "ey", "gxy", "sx", "sy", "txy"], rows)

    def _selected_ids_from_table(self, table: QTableWidget) -> set[int]:
        selected: set[int] = set()
        selection_model = table.selectionModel()
        if selection_model is None:
            return selected
        for idx in selection_model.selectedRows(0):
            item = table.item(idx.row(), 0)
            if item is not None:
                try:
                    selected.add(int(item.text()))
                except ValueError:
                    pass
        return selected

    def _on_node_selection_changed(self) -> None:
        self._selected_node_ids = self._selected_ids_from_table(self._node_table)
        self._mesh_canvas.set_highlighted_nodes(self._selected_node_ids)

    def _on_element_selection_changed(self) -> None:
        self._selected_element_ids = self._selected_ids_from_table(self._element_table)
        self._mesh_canvas.set_highlighted_elements(self._selected_element_ids)

    def _on_canvas_node_picked(self, node_id: int) -> None:
        self._selected_node_ids = {node_id}
        self._mesh_canvas.set_highlighted_nodes({node_id})
        if hasattr(self, "_load_point_node_spin"):
            self._load_point_node_spin.setValue(node_id)
        if hasattr(self, "_bc_point_node_spin"):
            self._bc_point_node_spin.setValue(node_id)
        if self._pending_pick_context == "load_point":
            self._pending_pick_context = None
            self._log(self._ui(f"已选择集中力作用点: 节点 {node_id}。", f"Selected load point node: {node_id}."))
        elif self._pending_pick_context == "bc_point":
            self._pending_pick_context = None
            self._log(self._ui(f"已选择约束点: 节点 {node_id}。", f"Selected BC point node: {node_id}."))

    def _on_canvas_sketch_face_picked(self, x: float, y: float) -> None:
        context = self._pending_pick_context
        if context == "geometry_point_target":
            point_id = self._resolve_or_create_geometry_point_at(float(x), float(y))
            if point_id is None:
                self._load_bc_set_hint.setText(
                    self._ui("未命中可用几何点/边/区域，请重试。", "No geometry point/edge/region hit. Try again.")
                )
                return
            self._selected_geometry_point_ids = {point_id}
            preferred_set_id = self._preferred_geometry_set_id_for_pick("point")
            set_id = self._upsert_geometry_set_from_geometry_ids(
                entity_type="point",
                geometry_ids=[point_id],
                preferred_name=self._ui("几何点集合", "Geometry Point Set"),
                preferred_set_id=preferred_set_id,
            )
            if set_id is not None:
                self._last_geometry_pick_set_id = set_id
                idx = self._load_bc_target_set_combo.findData(set_id)
                if idx >= 0:
                    self._load_bc_target_set_combo.setCurrentIndex(idx)
            self._pending_pick_context = None
            self._set_canvas_tool("select")
            self._log(self._ui(f"已选择几何点 {point_id}。", f"Geometry point selected: {point_id}."))
            return
        if context == "geometry_point_set_multi":
            point_id = self._resolve_or_create_geometry_point_at(float(x), float(y))
            if point_id is None:
                self._load_bc_set_hint.setText(
                    self._ui("未命中可用几何点/边/区域，请重试。", "No geometry point/edge/region hit. Try again.")
                )
                return
            if point_id in self._selected_geometry_point_ids:
                self._selected_geometry_point_ids.remove(point_id)
            else:
                self._selected_geometry_point_ids.add(point_id)
            preferred_set_id = self._preferred_geometry_set_id_for_pick("point")
            set_id = self._upsert_geometry_set_from_geometry_ids(
                entity_type="point",
                geometry_ids=sorted(self._selected_geometry_point_ids),
                preferred_name=self._ui("几何点集合", "Geometry Point Set"),
                preferred_set_id=preferred_set_id,
            )
            if set_id is not None:
                self._last_geometry_pick_set_id = set_id
                idx = self._load_bc_target_set_combo.findData(set_id)
                if idx >= 0:
                    self._load_bc_target_set_combo.setCurrentIndex(idx)
            self._sync_load_bc_geometry_overlay()
            self._load_bc_set_hint.setText(
                self._ui(
                    f"连续点集拾取中：已选 {len(self._selected_geometry_point_ids)} 个点（Esc 结束）。",
                    f"Continuous point-set picking: {len(self._selected_geometry_point_ids)} point(s) selected (Esc to finish).",
                )
            )
            return
        if context == "geometry_split_edge_point":
            point_id = self._create_geometry_edge_point_at(float(x), float(y))
            if point_id is None:
                self._load_bc_set_hint.setText(
                    self._ui("未命中可用几何边，请重试。", "No geometry edge hit. Try again.")
                )
                return
            self._selected_geometry_point_ids.add(point_id)
            preferred_set_id = self._preferred_geometry_set_id_for_pick("point")
            set_id = self._upsert_geometry_set_from_geometry_ids(
                entity_type="point",
                geometry_ids=sorted(self._selected_geometry_point_ids),
                preferred_name=self._ui("几何点集合", "Geometry Point Set"),
                preferred_set_id=preferred_set_id,
            )
            if set_id is not None:
                self._last_geometry_pick_set_id = set_id
                idx = self._load_bc_target_set_combo.findData(set_id)
                if idx >= 0:
                    self._load_bc_target_set_combo.setCurrentIndex(idx)
            self._sync_load_bc_geometry_overlay()
            self._load_bc_set_hint.setText(
                self._ui(
                    f"边上插点已创建：当前点集 {len(self._selected_geometry_point_ids)} 个点（Esc 结束）。",
                    f"Edge split point created: point-set now has {len(self._selected_geometry_point_ids)} point(s) (Esc to finish).",
                )
            )
            return
        if context == "geometry_edge_target":
            tolerance = max(float(self._active_sketch_grid_step) * 0.25, 1e-4)
            edge_id, _ = self._pick_geometry_edge_id_at(float(x), float(y), tolerance=tolerance)
            if edge_id is None:
                self._load_bc_set_hint.setText(
                    self._ui("未命中可用几何边，请重试。", "No geometry edge hit. Try again.")
                )
                return
            self._selected_geometry_edge_ids = {edge_id}
            preferred_set_id = self._preferred_geometry_set_id_for_pick("edge")
            set_id = self._upsert_geometry_set_from_geometry_ids(
                entity_type="edge",
                geometry_ids=[edge_id],
                preferred_name=self._ui("几何边集合", "Geometry Edge Set"),
                preferred_set_id=preferred_set_id,
            )
            if set_id is not None:
                self._last_geometry_pick_set_id = set_id
                idx = self._load_bc_target_set_combo.findData(set_id)
                if idx >= 0:
                    self._load_bc_target_set_combo.setCurrentIndex(idx)
            self._pending_pick_context = None
            self._set_canvas_tool("select")
            self._log(self._ui(f"已选择几何边 {edge_id}。", f"Geometry edge selected: {edge_id}."))
            return
        if context == "mesh_edge_seed":
            tolerance = max(float(self._active_sketch_grid_step) * 0.25, 1e-4)
            edge_id, _ = self._pick_geometry_edge_id_at(float(x), float(y), tolerance=tolerance)
            if edge_id is None:
                self._mesh_selection_hint.setText(
                    self._ui("未命中可用几何边，请重试。", "No geometry edge hit. Try again.")
                )
                return
            if edge_id in self._selected_geometry_edge_ids:
                self._selected_geometry_edge_ids.remove(edge_id)
            else:
                self._selected_geometry_edge_ids.add(edge_id)
            self._update_mesh_selection_hint()
            self._sync_load_bc_geometry_overlay()
            self._log(self._ui(f"边种子目标切换: {edge_id}", f"Edge seed target toggled: {edge_id}"))
            return
        if context not in {"material_face_sketch", "component_face_sketch", "mesh_region_sketch", "load_bc_region_sketch"}:
            return
        region = self._pick_sketch_face_region_at(float(x), float(y))
        if region is None:
            if context == "material_face_sketch":
                self._material_face_hint.setText(
                    self._ui("未命中草图封闭面，请重试", "No closed sketch face hit. Try again.")
                )
            elif context == "component_face_sketch":
                self._component_face_hint.setText(
                    self._ui("未命中草图封闭面，请重试", "No closed sketch face hit. Try again.")
                )
            elif context == "load_bc_region_sketch":
                self._load_bc_set_hint.setText(
                    self._ui("未命中草图封闭面，请重试。", "No closed sketch face hit. Try again.")
                )
            else:
                self._mesh_selection_hint.setText(
                    self._ui("未命中草图封闭面，请重试", "No closed sketch face hit. Try again.")
                )
            return
        key = str(region.get("key"))
        label = str(region.get("label", key))
        added, label = self._toggle_sketch_region_selection(key)
        if context == "material_face_sketch":
            self._material_assign_elements.setText("")
            self._update_material_region_selection_hint()
        elif context == "component_face_sketch":
            self._update_component_region_selection_hint()
        elif context == "load_bc_region_sketch":
            region_count = len(self._selected_scene_region_ids_for_component_assignment())
            self._load_bc_set_hint.setText(
                self._ui(
                    f"区域拾取中：已选 {region_count} 个区域；点击“由选区创建区域集合”后可施加区域体力/自重。",
                    f"Region picking: {region_count} region(s) selected; capture a region set before applying body force/gravity.",
                )
            )
        else:
            self._update_mesh_selection_hint()
        self._log(
            self._ui(
                f"{'已加入' if added else '已移除'}草图面区域：{label}。",
                f"{'Added' if added else 'Removed'} sketch face region: {label}.",
            )
        )

    def _on_canvas_element_picked(self, element_id: int) -> None:
        context = self._pending_pick_context
        if context in {"material_face", "component_face", "mesh_region", "load_bc_region"}:
            region_ids = self._collect_connected_element_region(element_id)
            if not region_ids:
                region_ids = {element_id}
            added, picked_count, total_count = self._toggle_element_region_selection(region_ids)
            if context == "material_face":
                self._material_assign_elements.setText("")
                self._update_material_region_selection_hint()
            elif context == "component_face":
                self._update_component_region_selection_hint()
            elif context == "load_bc_region":
                region_count = len(self._selected_scene_region_ids_for_component_assignment())
                self._load_bc_set_hint.setText(
                    self._ui(
                        f"区域拾取中：本次 {picked_count} 个单元，当前映射 {region_count} 个区域；点击“由选区创建区域集合”。",
                        f"Region picking: {picked_count} elements this pick, mapped to {region_count} region(s); capture a region set.",
                    )
                )
            else:
                self._update_mesh_selection_hint()
            self._log(
                self._ui(
                    f"{'已加入' if added else '已移除'}连通区域：本次 {picked_count} 个单元，当前累计 {total_count} 个单元。",
                    f"{'Added' if added else 'Removed'} connected region: {picked_count} elements this pick, {total_count} selected in total.",
                )
            )
            return

        self._selected_element_ids = {element_id}
        self._selected_face_region_element_ids.clear()
        self._selected_face_region_sketch_keys.clear()
        self._scene_selection.active_region_ids.clear()
        self._mesh_canvas.set_highlighted_elements({element_id})
        self._mesh_canvas.set_highlighted_sketch_faces([])

    def _update_mapping_preview(self, entries: list[PhysicalGroupMappingEntry]) -> None:
        self._mapping_table.setSortingEnabled(False)
        self._mapping_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = [str(entry.dimension), str(entry.tag), entry.name, entry.mapped_to, entry.status, str(entry.item_count), entry.note]
            background = mapping_row_color(entry.status)
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setBackground(background)
                self._mapping_table.setItem(row, col, item)
        self._mapping_table.setSortingEnabled(True)

        counts = summarize_mapping_entries(entries)
        self._mapping_total_badge.setText(f"total {counts['total_group_count']}")
        self._mapping_mapped_badge.setText(f"mapped {counts['mapped_group_count']}")
        self._mapping_warning_badge.setText(f"warning {counts['warning_group_count']}")
        self._mapping_unrec_badge.setText(f"unrecognized {counts['unrecognized_group_count']}")
        self._mapping_summary.setText(f"groups: total={counts['total_group_count']}, mapped={counts['mapped_group_count']}, warning={counts['warning_group_count']}, unrecognized={counts['unrecognized_group_count']}")
        self._mapping_risk_text.setText("No physical groups detected." if counts["total_group_count"] == 0 else "Review mapping statuses before solve.")
        self._btn_apply_mapping.setEnabled(self._model is not None and self._last_gmsh_import is not None and not self._mapping_applied)

    def _format_summary(self, result: StaticSolveResult, backend: str) -> str:
        assert self._model is not None
        status = summarize_model_state(self._model, backend=backend)
        return (
            f"backend = {status['backend']}\n"
            f"nodes = {result.summary.node_count}\n"
            f"elements = {result.summary.element_count}\n"
            f"materials = {status['material_count']}\n"
            f"boundary_conditions = {status['boundary_condition_count']}\n"
            f"loads = {status['load_count']}\n\n"
            f"total_dof = {result.summary.total_dof}\n"
            f"max_displacement = {result.summary.max_displacement:.6e}\n"
            f"displacement_vector_size = {result.displacements.size}\n"
            f"reaction_vector_size = {result.reactions.size}"
        )

    def _update_status_labels(self) -> None:
        backend = self._backend_combo.currentText()
        self._status_backend.setText(backend)
        self._mesh_canvas.set_active_backend(backend)
        if self._model is None:
            self._status_node_count.setText("0")
            self._status_element_count.setText("0")
            self._status_material_count.setText("0")
            self._status_bc_count.setText("0")
            self._status_load_count.setText("0")
            self._material_name_to_id.clear()
            self._material_id_to_name.clear()
            self._material_assign_id_spin.setValue(1)
            self._material_name_edit.setText("Soil-1")
            self._sync_material_controls()
            return
        status = summarize_model_state(self._model, backend=backend)
        self._status_node_count.setText(str(status["node_count"]))
        self._status_element_count.setText(str(status["element_count"]))
        self._status_material_count.setText(str(status["material_count"]))
        self._status_bc_count.setText(str(status["boundary_condition_count"]))
        self._status_load_count.setText(str(status["load_count"]))
        current_material_ids = {item.id for item in self._model.materials}
        self._material_id_to_name = {
            mid: name
            for mid, name in self._material_id_to_name.items()
            if mid in current_material_ids
        }
        for mid in sorted(current_material_ids):
            if mid not in self._material_id_to_name:
                self._material_id_to_name[mid] = f"MAT-{mid}"
        self._material_name_to_id = {name: mid for mid, name in self._material_id_to_name.items()}
        if current_material_ids:
            max_mid = max(current_material_ids)
            if self._material_assign_id_spin.value() not in current_material_ids:
                self._material_assign_id_spin.setValue(max_mid)
        self._sync_material_controls()

    def _on_backend_changed(self, backend: str) -> None:
        if not hasattr(self, "_status_backend"):
            return
        self._status_backend.setText(backend)
        self._mesh_canvas.set_active_backend(backend)
        self._update_status_labels()
        self._mark_results_stale("Backend changed")

    def _on_contour_variable_changed(self, _: int) -> None:
        self._mesh_canvas.set_contour_variable(str(self._contour_combo.currentData()))

    def _on_language_changed(self, _: int) -> None:
        if self._is_retranslating:
            return
        language = self._language_combo.currentData()
        if not isinstance(language, str):
            return
        self._language = normalize_language(language)
        save_language(self._language, self._settings)
        self._mesh_canvas.set_language(self._language)
        self._retranslate_ui()

    def _update_model_tree(self) -> None:
        self._tree.clear()
        if self._model is None:
            return
        root = QTreeWidgetItem(self._tree, ["Model"])
        mesh_node = QTreeWidgetItem(root, ["Mesh"])
        QTreeWidgetItem(mesh_node, [f"Nodes: {len(self._model.mesh.nodes)}"])
        QTreeWidgetItem(mesh_node, [f"Elements: {len(self._model.mesh.elements)}"])
        material_counts = self._material_usage_counts()
        material_node = QTreeWidgetItem(root, [f"Materials: {len(self._model.materials)}"])
        for material in sorted(self._model.materials, key=lambda item: int(item.id)):
            name = self._material_display_name(material.id)
            usage_count = material_counts.get(int(material.id), 0)
            QTreeWidgetItem(
                material_node,
                [
                    f"{name} [ID={material.id}] | E={float(material.young_modulus):.3e} | nu={float(material.poisson_ratio):.3f} | {self._material_mode_text(material)} | elems={usage_count}",
                ],
            )
        QTreeWidgetItem(root, [f"Boundary Conditions: {len(self._model.boundary_conditions)}"])
        QTreeWidgetItem(root, [f"Loads: {len(self._model.loads)}"])
        if self._scene_project.load_definitions or self._scene_project.boundary_definitions:
            load_bc_node = QTreeWidgetItem(root, [self._ui("几何载荷/约束定义", "Geometry Load/BC Definitions")])
            QTreeWidgetItem(load_bc_node, [self._ui(f"载荷定义: {len(self._scene_project.load_definitions)}", f"Load definitions: {len(self._scene_project.load_definitions)}")])
            QTreeWidgetItem(load_bc_node, [self._ui(f"约束定义: {len(self._scene_project.boundary_definitions)}", f"BC definitions: {len(self._scene_project.boundary_definitions)}")])
            load_bc_node.setExpanded(True)
        root.setExpanded(True)
        mesh_node.setExpanded(True)
        material_node.setExpanded(True)

    def _retranslate_ui(self) -> None:
        self._is_retranslating = True
        try:
            self.setWindowTitle(self._tr("app.title", "RockFEM - 2D AI FEM Workbench"))
            self._brand_title_label.setText("RockFEM")
            self._brand_subtitle_label.setText(self._tr("brand.subtitle", "AI Assisted 2D Geotech & Structural FEM"))

            self._menu_file.setTitle(self._tr("menu.file", "File"))
            self._menu_part.setTitle(self._tr("menu.part", "Part"))
            self._menu_material.setTitle(self._tr("menu.material", "Material"))
            self._menu_tool.setTitle(self._tr("menu.tool", "Tool"))
            self._menu_view.setTitle(self._tr("menu.view", "View"))
            self._menu_visualization.setTitle(self._tr("menu.visualization", "Visualization"))
            self._menu_help.setTitle(self._tr("menu.help", "Help"))
            self._menu_language.setTitle(self._tr("menu.language", "Language"))

            self._action_file_new.setText(self._tr("action.file.new", "New Model"))
            self._action_file_import.setText(self._tr("action.file.import", "Import Mesh"))
            self._action_file_export.setText(self._tr("action.file.export", "Export Results"))
            self._action_file_exit.setText(self._tr("action.file.exit", "Exit"))
            self._action_part_jump.setText(self._tr("action.part.jump", "Go To Part Builder"))
            self._action_part_build.setText(self._tr("action.part.build", "Build Closed Sketch"))
            self._action_material_new.setText(self._tr("action.material.new", "Create Isotropic Material"))
            self._action_material_assign.setText(self._tr("action.material.assign", "Assign To Region"))
            self._action_tool_export_canvas.setText(self._tr("action.tool.export_canvas", "Export View Image"))
            self._action_tool_layout_transform.setText(self._tr("action.tool.layout_transform", "Layout Transform"))
            self._action_tool_toggle_log.setText(self._tr("action.tool.log", "Toggle Log Panel"))
            self._action_view_workspace.setText(self._tr("action.view.workspace", "Open View Workspace"))
            self._action_view_results.setText(self._tr("action.view.results", "Open Results Workspace"))
            self._action_visualization_disp.setText(self._tr("action.viz.disp", "Displacement Contour"))
            self._action_visualization_mises.setText(self._tr("action.viz.mises", "Mises Contour"))
            self._action_visualization_principal.setText(self._tr("action.viz.principal", "Max Principal Stress"))
            self._action_help_about.setText(self._tr("action.help.about", "About RockFEM"))
            self._action_help_workflow.setText(self._tr("action.help.workflow", "Workflow Guide"))
            self._action_lang_zh.setText(self._tr("language.zh", "中文"))
            self._action_lang_en.setText(self._tr("language.en", "English"))
            self._action_lang_zh.setChecked(self._language == "zh_CN")
            self._action_lang_en.setChecked(self._language == "en_US")

            self._workflow_buttons["model"].setText(self._tr("workflow.model", "Model"))
            self._workflow_buttons["part"].setText(self._tr("workflow.part", "Part"))
            self._workflow_buttons["material"].setText(self._tr("workflow.material", "Material"))
            self._workflow_buttons["assembly"].setText(self._tr("workflow.assembly", "Assembly/Component"))
            self._workflow_buttons["load_bc"].setText(self._tr("workflow.load_bc", "Load/BC"))
            self._workflow_buttons["mesh"].setText(self._tr("workflow.mesh", "Mesh"))
            self._workflow_buttons["job"].setText(self._tr("workflow.job", "Job"))
            self._workflow_buttons["visualization"].setText(self._tr("workflow.visualization", "Visualization"))
            self._workflow_buttons["report"].setText(self._tr("workflow.report", "Report"))
            self._btn_quick_undo.setToolTip(self._tr("quick.undo", "撤销部件操作 (Ctrl+Z)"))
            self._btn_quick_redo.setToolTip(self._tr("quick.redo", "重做部件操作 (Ctrl+Y)"))
            self._btn_quick_zoom_in.setToolTip(self._tr("quick.zoom_in", "放大视图"))
            self._btn_quick_zoom_out.setToolTip(self._tr("quick.zoom_out", "缩小视图"))
            self._btn_quick_reset_view.setToolTip(self._tr("quick.reset_view", "重置视图"))
            self._btn_quick_cancel.setToolTip(self._tr("quick.cancel", "取消当前绘制/拾取 (Esc)"))

            self._btn_part_tool_select.setToolTip(self._tr("partstrip.select", "选择/拖拽"))
            self._btn_part_tool_add_point.setToolTip(self._tr("partstrip.add_point", "画布点录入"))
            self._btn_part_tool_measure_len.setToolTip(self._tr("partstrip.measure_len", "测量长度"))
            self._btn_part_tool_measure_ang.setToolTip(self._tr("partstrip.measure_ang", "测量角度"))
            self._btn_part_tool_rect.setToolTip(self._tr("partstrip.rect", "矩形拖拽绘制"))
            self._btn_part_tool_circle.setToolTip(self._tr("partstrip.circle", "圆形拖拽绘制"))
            self._btn_part_tool_ellipse.setToolTip(self._tr("partstrip.ellipse", "椭圆拖拽绘制"))
            self._btn_part_tool_close.setToolTip(self._tr("partstrip.close", "闭合草图"))
            self._btn_part_tool_clear_selected.setToolTip(self._tr("partstrip.clear_selected", "清除选中线段/图形"))
            self._btn_part_tool_clear_all.setToolTip(self._tr("partstrip.clear_all", "清除全部图形"))
            self._part_shape_w_label.setText(self._tr("partstrip.width", "宽"))
            self._part_shape_h_label.setText(self._tr("partstrip.height", "高"))
            self._part_shape_w.setToolTip(self._tr("partstrip.width.tip", "图元宽度"))
            self._part_shape_h.setToolTip(self._tr("partstrip.height.tip", "图元高度"))
            self._part_shape_lock_ratio.setText(self._tr("partstrip.lock_ratio", "锁比例"))
            self._part_shape_lock_ratio.setToolTip(self._tr("partstrip.lock_ratio.tip", "锁定宽高比例，修改一边自动联动另一边"))
            self._btn_part_quick_apply.setToolTip(self._tr("partstrip.apply_dim", "应用新尺寸"))
            self._part_quick_mode.setItemText(0, self._tr("partstrip.mode.length", "长度"))
            self._part_quick_mode.setItemText(1, self._tr("partstrip.mode.angle", "角度"))
            self._part_quick_mode.setToolTip(self._tr("partstrip.mode.tip", "选择新尺寸类型"))
            self._part_quick_value.setToolTip(self._tr("partstrip.value.tip", "输入新尺寸数值并应用"))
            self._material_toolbar_label.setText(self._tr("material.strip.label", "当前材料"))
            self._material_toolbar_template_combo.setToolTip(self._tr("material.strip.template_combo", "常见岩土材料模板"))
            self._btn_material_tool_apply_template.setToolTip(self._tr("material.strip.template", "将模板参数写入当前材料表单"))
            self._btn_material_tool_create.setToolTip(self._tr("material.strip.create", "创建或更新当前材料"))
            self._btn_material_tool_rename.setToolTip(self._tr("material.strip.rename", "重命名当前材料"))
            self._btn_material_tool_duplicate.setToolTip(self._tr("material.strip.duplicate", "复制当前材料并微调参数"))
            self._btn_material_tool_pick_face.setToolTip(self._tr("material.strip.pick", "选择要分配的区域"))
            self._btn_material_tool_assign.setToolTip(self._tr("material.strip.assign", "将当前材料分配到所选区域"))
            self._btn_material_tool_delete.setToolTip(self._tr("material.strip.delete", "删除当前材料（需未被单元使用）"))
            self._load_bc_command_label.setText(self._tr("loadbc.strip.label", "载荷/约束工具"))
            self._btn_loadbc_tool_create_load.setText(self._tr("loadbc.strip.create_load", "创建载荷"))
            self._btn_loadbc_tool_load_manager.setText(self._tr("loadbc.strip.load_manager", "载荷管理器"))
            self._btn_loadbc_tool_create_bc.setText(self._tr("loadbc.strip.create_bc", "创建边界"))
            self._btn_loadbc_tool_bc_manager.setText(self._tr("loadbc.strip.bc_manager", "边界管理器"))
            self._btn_loadbc_tool_pick_point.setText(self._tr("loadbc.strip.pick_point", "选点"))
            self._btn_loadbc_tool_pick_edge.setText(self._tr("loadbc.strip.pick_edge", "选边"))
            self._btn_loadbc_tool_finish.setText(self._tr("loadbc.strip.finish", "完成"))
            self._btn_loadbc_tool_cancel.setText(self._tr("loadbc.strip.cancel", "取消"))
            self._btn_loadbc_tool_create_load.setToolTip(self._tr("loadbc.strip.create_load.tip", "打开 Abaqus 式创建载荷流程"))
            self._btn_loadbc_tool_load_manager.setToolTip(self._tr("loadbc.strip.load_manager.tip", "打开独立载荷管理器：编辑、复制、重命名、删除、抑制、定位"))
            self._btn_loadbc_tool_create_bc.setToolTip(self._tr("loadbc.strip.create_bc.tip", "打开二维边界条件创建流程"))
            self._btn_loadbc_tool_bc_manager.setToolTip(self._tr("loadbc.strip.bc_manager.tip", "打开独立边界条件管理器：编辑、复制、重命名、删除、抑制、定位"))
            self._btn_loadbc_tool_pick_point.setToolTip(self._tr("loadbc.strip.pick_point.tip", "在画布上选择几何点或创建保留点"))
            self._btn_loadbc_tool_pick_edge.setToolTip(self._tr("loadbc.strip.pick_edge.tip", "在画布上选择几何边，用于边载荷或边约束"))
            self._btn_loadbc_tool_finish.setToolTip(self._tr("loadbc.strip.finish.tip", "完成当前 Load/BC 几何选择"))
            self._btn_loadbc_tool_cancel.setToolTip(self._tr("loadbc.strip.cancel.tip", "取消当前交互命令"))
            self._btn_material_apply_template.setToolTip(self._tr("material.strip.template", "将模板参数写入当前材料表单"))
            self._material_action_template_label.setText(self._tr("rock.material.action.template", "材料工具"))
            self._material_action_template_combo.setToolTip(self._tr("material.strip.template_combo", "常见岩土材料模板"))
            self._btn_material_action_apply_template.setToolTip(self._tr("material.strip.template", "将模板参数写入当前材料表单"))
            self._btn_material_action_create.setToolTip(self._tr("material.strip.create", "创建或更新当前材料"))
            self._btn_material_action_rename.setToolTip(self._tr("material.strip.rename", "重命名当前材料"))
            self._btn_material_action_duplicate.setToolTip(self._tr("material.strip.duplicate", "复制当前材料并微调参数"))
            self._btn_material_action_delete.setToolTip(self._tr("material.strip.delete", "删除当前材料（需未被单元使用）"))
            self._btn_material_action_pick_face.setToolTip(self._tr("material.strip.pick", "选择要分配的区域"))
            self._btn_material_action_assign.setToolTip(self._tr("material.strip.assign", "将当前材料分配到所选区域"))
            self._refresh_part_sketch_table()

            self._btn_toolbar_demo.setText(self._tr("toolbar.load_demo", "Load Demo"))
            self._btn_toolbar_geotech.setText(self._tr("toolbar.geotech", "Geotech"))
            self._btn_toolbar_import.setText(self._tr("toolbar.import", "Import"))
            self._btn_toolbar_solve.setText(self._tr("toolbar.solve", "Solve"))
            self._btn_toolbar_export.setText(self._tr("toolbar.export_results", "Export Results"))
            self._btn_toolbar_demo.setToolTip(self._tr("toolbar.tip.demo", "加载演示模型"))
            self._btn_toolbar_geotech.setToolTip(self._tr("toolbar.tip.geotech", "运行岩土模板入口"))
            self._btn_toolbar_import.setToolTip(self._tr("toolbar.tip.import", "导入网格/模型文件"))
            self._btn_toolbar_solve.setToolTip(self._tr("toolbar.tip.solve", "提交当前模型求解"))
            self._btn_toolbar_export.setToolTip(self._tr("toolbar.tip.export", "导出当前结果"))
            self._btn_toggle_log.setToolTip(self._tr("toolbar.tip.log", "显示或隐藏日志"))
            self._backend_label.setText(self._tr("toolbar.backend", "Backend"))
            self._language_label.setText(self._tr("toolbar.language", "Language"))
            self._btn_toggle_log.setText(self._tr("toolbar.log", "Log"))
            self._action_import_json.setText(self._tr("toolbar.import_json", "Import T3 JSON Mesh"))
            self._action_import_msh.setText(self._tr("toolbar.import_gmsh", "Import Gmsh .msh"))
            self._action_geotech_quick.setText(self._tr("geotech.action.quick", "Quick Geotech Template"))
            self._action_geotech_from_text.setText(self._tr("geotech.action.text", "Geotech from Text..."))
            self._action_geotech_from_csv.setText(self._tr("geotech.action.csv", "Geotech from CSV..."))

            langs = available_languages()
            self._language_combo.blockSignals(True)
            self._language_combo.clear()
            self._language_combo.addItem(langs["zh_CN"], "zh_CN")
            self._language_combo.addItem(langs["en_US"], "en_US")
            self._language_combo.setCurrentIndex(0 if self._language == "zh_CN" else 1)
            self._language_combo.blockSignals(False)

            self._side_tabs.setTabText(0, self._tr("tab.model", "Model"))
            self._side_tabs.setTabText(1, self._tr("tab.mapping", "Mapping"))
            self._side_tabs.setTabText(2, self._tr("tab.view", "View"))
            self._side_tabs.setTabText(3, self._tr("tab.results", "Results"))
            self._side_tabs.setTabText(4, self._tr("tab.log", "Log"))

            self._status_group.setTitle(self._tr("group.model_status", "Model Status"))
            self._runtime_group.setTitle(self._tr("group.solve_state", "Solve State"))
            self._rock_model_group.setTitle(self._tr("group.rock_model", "模型创建 / Model Creation"))
            self._rock_part_group.setTitle(self._tr("group.rock_part", "部件创建 / Part Sketch"))
            self._rock_material_group.setTitle(self._tr("group.rock_material", "材料属性 / Material"))
            self._material_manager_group.setTitle(self._tr("group.material_manager", "材料管理器 / Material Manager"))
            self._rock_assembly_group.setTitle(self._tr("group.rock_assembly", "装配与组件 / Assembly & Component"))
            self._rock_load_bc_group.setTitle(self._tr("group.rock_load_bc", "载荷与约束 / Load & BC"))
            self._rock_mesh_group.setTitle(self._tr("group.rock_mesh", "网格 / Mesh"))
            self._rock_job_group.setTitle(self._tr("group.rock_job", "提交作业 / Job"))
            self._rock_report_group.setTitle(self._tr("group.rock_report", "报告 / Report"))
            self._geotech_group.setTitle(self._tr("group.geotech", "Geotechnical Template"))
            self._explorer_group.setTitle(self._tr("group.model_explorer", "Model Explorer"))
            self._mapping_summary_group.setTitle(self._tr("group.mapping_summary", "Physical Group Mapping Summary"))
            self._mapping_table_group.setTitle(self._tr("group.mapping_details", "Group Details"))
            self._display_group.setTitle(self._tr("group.display", "Display"))
            self._result_display_group.setTitle(self._tr("group.results_display", "Result Display"))

            self._lbl_nodes.setText(self._tr("label.nodes", "Nodes"))
            self._lbl_elements.setText(self._tr("label.elements", "Elements"))
            self._lbl_materials.setText(self._tr("label.materials", "Materials"))
            self._lbl_bc.setText(self._tr("label.bc", "BC"))
            self._lbl_loads.setText(self._tr("label.loads", "Loads"))
            self._lbl_backend.setText(self._tr("label.backend", "Backend"))

            self._rock_model_intro.setText(
                self._tr(
                    "rock.model.intro",
                    "支持 AI 识别图片入口、AI 工程语境解析，以及普通二维模型快速创建。",
                )
            )
            self._btn_rock_ai_image.setText(self._tr("rock.model.ai_image", "AI识别图片入口"))
            self._btn_rock_ai_context.setText(self._tr("rock.model.ai_context", "基于工程语境生成模型"))
            self._btn_rock_create_2d.setText(self._tr("rock.model.create_2d", "普通二维模型创建"))
            self._rock_ai_image_path.setPlaceholderText(self._tr("rock.model.ai_image.path", "AI 图像路径"))
            self._rock_ai_context_text.setPlaceholderText(
                self._tr(
                    "rock.model.context.placeholder",
                    "输入工程语境，例如：边坡分层、基础埋深、荷载等。",
                )
            )
            self._rock_model_width_label.setText(self._tr("rock.model.width", "Width"))
            self._rock_model_height_label.setText(self._tr("rock.model.height", "Height"))
            self._rock_model_seed_label.setText(self._tr("rock.model.grid_step", "Sketch Grid Step"))
            self._btn_model_measure.setText(self._tr("rock.model.measure", "测量距离"))
            self._btn_model_reset_view.setText(self._tr("rock.model.reset_view", "重置视图"))
            if not self._model_measure_result.text().strip():
                self._model_measure_result.setText(self._tr("rock.model.measure.wait", "测距: 等待选择两点"))

            self._rock_part_intro.setText(
                self._tr(
                    "rock.part.intro",
                    "通过坐标或画布点击录入点，形成封闭线段并生成部件。当前提供三角扇分区。",
                )
            )
            self._btn_part_add_point.setText(self._tr("rock.part.add_point", "新增点"))
            self._part_capture_from_canvas.setText(self._tr("rock.part.capture", "启用画布点录入"))
            self._part_snap_to_grid.setText(self._tr("rock.part.snap_grid", "点吸附到网格"))
            self._btn_part_clear_selected.setText(self._tr("partstrip.clear_selected", "清除选中线段/图形"))
            self._btn_part_clear.setText(self._tr("rock.part.clear", "清空草图"))
            self._btn_part_close.setText(self._tr("rock.part.close_loop", "闭合草图"))
            self._btn_part_build.setText(self._tr("rock.part.build", "封闭草图生成部件"))
            self._part_toolbox_group.setTitle(self._tr("rock.part.toolbox", "草图标注工具箱"))
            self._btn_part_measure.setText(self._tr("rock.part.measure", "测量距离"))
            self._btn_part_measure_fill_length.setText(self._tr("rock.part.fill_measure", "测距填入长度"))
            self._part_length_target_label.setText(self._tr("rock.part.length_target", "目标长度"))
            self._btn_part_apply_length.setText(self._tr("rock.part.apply_length", "应用长度标定"))
            self._part_angle_target_label.setText(self._tr("rock.part.angle_target", "目标角度(°)"))
            self._btn_part_apply_angle.setText(self._tr("rock.part.apply_angle", "应用角度标定"))
            self._btn_part_constraint_horizontal.setText(self._tr("rock.part.constraint.horizontal", "水平"))
            self._btn_part_constraint_vertical.setText(self._tr("rock.part.constraint.vertical", "垂直"))
            self._btn_part_constraint_collinear.setText(self._tr("rock.part.constraint.collinear", "共线"))
            self._btn_part_constraint_parallel.setText(self._tr("rock.part.constraint.parallel", "平行"))
            self._btn_part_constraint_perpendicular.setText(self._tr("rock.part.constraint.perpendicular", "垂直于"))
            self._btn_part_constraint_horizontal.setToolTip(
                self._tr("rock.part.constraint.horizontal.tip", "将选中线段约束为水平（需要选中1条线段）。")
            )
            self._btn_part_constraint_vertical.setToolTip(
                self._tr("rock.part.constraint.vertical.tip", "将选中线段约束为垂直（需要选中1条线段）。")
            )
            self._btn_part_constraint_collinear.setToolTip(
                self._tr("rock.part.constraint.collinear.tip", "将3个选中点约束为共线。")
            )
            self._btn_part_constraint_parallel.setToolTip(
                self._tr("rock.part.constraint.parallel.tip", "将两条选中线段约束为平行（需要4点）。")
            )
            self._btn_part_constraint_perpendicular.setToolTip(
                self._tr("rock.part.constraint.perpendicular.tip", "将两条选中线段约束为垂直（需要4点）。")
            )
            if not self._part_dimension_status.text().strip():
                self._part_dimension_status.setText(self._tr("rock.part.dimension.hint", "提示：长度标定选2点，角度标定按 A-顶点B-C 选3点。"))

            self._workflow_buttons["model"].setToolTip(self._tr("workflow.tip.model", "模型创建：AI识别/几何范围"))
            self._workflow_buttons["part"].setToolTip(self._tr("workflow.tip.part", "部件草图：录入点、闭合轮廓、几何标定"))
            self._workflow_buttons["material"].setToolTip(self._tr("workflow.tip.material", "材料定义与区域分配"))
            self._workflow_buttons["assembly"].setToolTip(self._tr("workflow.tip.assembly", "组件管理与可见性过滤"))
            self._workflow_buttons["load_bc"].setToolTip(self._tr("workflow.tip.loadbc", "载荷与约束设置"))
            self._workflow_buttons["mesh"].setToolTip(self._tr("workflow.tip.mesh", "网格参数与生成算法"))
            self._workflow_buttons["job"].setToolTip(self._tr("workflow.tip.job", "作业提交与求解"))
            self._workflow_buttons["visualization"].setToolTip(self._tr("workflow.tip.viz", "结果云图与查看"))
            self._workflow_buttons["report"].setToolTip(self._tr("workflow.tip.report", "分析报告输出"))

            self._rock_material_intro.setText(
                self._tr(
                    "rock.material.intro",
                    "当前优先支持各向同性弹性材料；可新建材料并分配到区域（单元集合）。",
                )
            )
            self._material_name_label.setText(self._tr("rock.material.name", "Material Name"))
            self._material_e_label.setText(self._tr("rock.material.e", "Elastic E"))
            self._material_nu_label.setText(self._tr("rock.material.nu", "Poisson nu"))
            self._material_plane_stress_label.setText(self._tr("rock.material.plane_stress", "Plane Stress"))
            self._material_unit_hint.setText(
                self._tr(
                    "rock.material.unit_hint",
                    "单位建议：E 使用 Pa，nu 无量纲。可用下方模板快速填入常见材料参数。",
                )
            )
            self._material_template_label.setText(self._tr("rock.material.template", "Template"))
            self._btn_material_apply_template.setText(self._tr("rock.material.template.apply", "应用模板"))
            self._material_assign_id_label.setText(self._tr("rock.material.assign_mid", "Assign Material"))
            self._material_assign_elements_label.setText(self._tr("rock.material.assign_region", "Selected Region"))
            self._material_assign_elements.setPlaceholderText(self._tr("rock.material.assign_region.placeholder", "auto (from picked face region) / optional element ids"))
            self._btn_material_create.setText(self._tr("rock.material.create", "创建/更新材料"))
            self._btn_material_rename.setText(self._tr("rock.material.rename", "重命名"))
            self._btn_material_duplicate.setText(self._tr("rock.material.duplicate", "复制"))
            self._btn_material_delete.setText(self._tr("rock.material.delete", "删除当前材料"))
            self._material_reassign_target_label.setText(self._tr("rock.material.reassign.target", "替换为"))
            self._btn_material_reassign_all.setText(self._tr("rock.material.reassign.apply", "批量重分配"))
            self._btn_material_pick_face.setText(self._tr("rock.material.pick_face", "选择面区域"))
            if not self._material_face_hint.text().strip():
                self._material_face_hint.setText(self._tr("rock.material.pick_face.hint", "未选择区域"))
            self._btn_material_assign.setText(self._tr("rock.material.assign", "分配到区域"))

            self._rock_assembly_intro.setText(
                self._tr(
                    "rock.assembly.intro",
                    "组件页用于按区域组织、隐藏/隔离以及统计；不改变全局求解模型。",
                )
            )
            self._component_manager_intro.setText(
                self._tr(
                    "rock.assembly.component_intro",
                    "可将已选区域（草图面或网格面）归属到组件，并按组件进行显示过滤与统计。",
                )
            )
            self._btn_component_pick_face.setText(self._tr("rock.assembly.component.pick", "选择区域"))
            if not self._component_face_hint.text().strip():
                self._component_face_hint.setText(self._tr("rock.assembly.component.pick.hint", "未选择区域"))
            self._btn_component_create_from_selected.setText(self._tr("rock.assembly.component.create", "由选区新建组件"))
            self._btn_component_rename.setText(self._tr("rock.assembly.component.rename", "重命名组件"))
            self._btn_component_toggle_visibility.setText(self._tr("rock.assembly.component.toggle", "显示/隐藏"))
            self._btn_component_isolate.setText(self._tr("rock.assembly.component.isolate", "隔离选中组件"))
            self._btn_component_show_all.setText(self._tr("rock.assembly.component.show_all", "显示全部组件"))
            self._assembly_transform_group.setTitle(self._tr("rock.assembly.transform_group", "布局变换 / Layout Transform (Legacy)"))
            self._assembly_transform_intro.setText(
                self._tr(
                    "rock.assembly.transform_intro",
                    "兼容旧流程：对当前实例做整体平移，不参与 Component 管理主流程。",
                )
            )
            self._assembly_instance_label.setText(self._tr("rock.assembly.instance", "Instance"))
            self._assembly_dx_label.setText(self._tr("rock.assembly.dx", "Offset dx"))
            self._assembly_dy_label.setText(self._tr("rock.assembly.dy", "Offset dy"))
            self._btn_apply_assembly_offset.setText(self._tr("rock.assembly.apply", "应用装配偏移"))

            self._rock_load_bc_intro.setText(
                self._tr(
                    "rock.loadbc.intro",
                    "载荷支持几何集中力、边线载/边压力、区域体力/自重；约束支持点/边目标与 UX/UY 控制，求解前统一解析到网格。",
                )
            )
            self._load_bc_set_group.setTitle(self._tr("rock.loadbc.set.group", "目标集合 / Target Set"))
            self._load_bc_target_mode_label.setText(self._tr("rock.loadbc.set.mode", "目标模式"))
            self._load_bc_target_mode_combo.setItemText(0, self._tr("rock.loadbc.set.mode.mesh", "网格目标"))
            self._load_bc_target_mode_combo.setItemText(1, self._tr("rock.loadbc.set.mode.geometry", "几何目标"))
            self._load_bc_target_set_label.setText(self._tr("rock.loadbc.set.target", "目标集合"))
            self._btn_load_bc_capture_set.setText(self._tr("rock.loadbc.set.capture", "从当前目标创建集合"))
            self._btn_load_bc_capture_region_set.setText(self._tr("rock.loadbc.set.capture_region", "由选区创建区域集合"))
            self._btn_load_bc_capture_component_set.setText(self._tr("rock.loadbc.set.capture_component", "由组件创建组件集合"))
            self._btn_load_bc_pick_region.setText(self._tr("rock.loadbc.set.pick_region", "画布选区域"))
            self._btn_load_bc_pick_geo_point.setText(self._tr("rock.loadbc.set.pick_geo_point", "几何选点"))
            self._btn_load_bc_pick_geo_edge.setText(self._tr("rock.loadbc.set.pick_geo_edge", "几何选边"))
            self._btn_load_bc_pick_geo_point_set.setText(self._tr("rock.loadbc.set.point_set_tool", "连续点集"))
            self._btn_load_bc_split_edge_point.setText(self._tr("rock.loadbc.set.split_edge_point", "边上插点"))
            self._btn_load_bc_create_point_by_coord.setText(self._tr("rock.loadbc.set.coord_point", "按坐标建点"))
            self._btn_load_bc_create_edge_fraction_point.setText(self._tr("rock.loadbc.set.edge_fraction_point", "边上比例点"))
            self._btn_load_bc_clear_point_set.setText(self._tr("rock.loadbc.set.clear_point_set", "清空点集"))
            self._btn_load_bc_finish_pick.setText(self._tr("rock.loadbc.set.finish_pick", "完成拾取"))
            self._btn_load_bc_cleanup_points.setText(self._tr("rock.loadbc.set.cleanup_points", "清理辅助点"))
            self._load_bc_manager_launch_group.setTitle(self._tr("rock.loadbc.manager.launch", "Abaqus 式管理"))
            self._btn_create_load_dialog.setText(self._tr("rock.loadbc.dialog.create_load", "创建载荷..."))
            self._btn_open_load_manager_dialog.setText(self._tr("rock.loadbc.dialog.load_manager", "载荷管理器..."))
            self._btn_create_bc_dialog.setText(self._tr("rock.loadbc.dialog.create_bc", "创建边界条件..."))
            self._btn_open_bc_manager_dialog.setText(self._tr("rock.loadbc.dialog.bc_manager", "边界条件管理器..."))
            self._load_bc_tool_group.setTitle(self._tr("rock.loadbc.tools.group", "创建工具 / Create Load"))
            self._btn_tool_point_force.setText(self._tr("rock.loadbc.tools.point_force", "集中力"))
            self._btn_tool_edge_load.setText(self._tr("rock.loadbc.tools.edge_load", "边线载"))
            self._btn_tool_edge_pressure.setText(self._tr("rock.loadbc.tools.edge_pressure", "边压力"))
            self._btn_tool_body_force.setText(self._tr("rock.loadbc.tools.body_force", "区域体力"))
            self._btn_tool_gravity.setText(self._tr("rock.loadbc.tools.gravity", "自重"))
            self._btn_add_point_load.setText(self._tr("rock.loadbc.add_point", "添加集中力"))
            self._load_dist_type_combo.setItemText(0, self._tr("rock.loadbc.edge.type.line", "边线载（总力）"))
            self._load_dist_type_combo.setItemText(1, self._tr("rock.loadbc.edge.type.pressure", "边压力（强度）"))
            self._load_dist_direction_mode_combo.setItemText(0, self._tr("rock.loadbc.edge.dir.vector", "自定义向量"))
            self._load_dist_direction_mode_combo.setItemText(1, self._tr("rock.loadbc.edge.dir.normal", "边法向"))
            self._load_dist_direction_mode_combo.setItemText(2, self._tr("rock.loadbc.edge.dir.reverse_normal", "反向法向"))
            self._btn_pick_dist_edge.setText(self._tr("rock.loadbc.pick_dist_edge", "画布选边"))
            self._btn_add_dist_load.setText(self._tr("rock.loadbc.add_dist", "添加边载荷"))
            self._region_load_group.setTitle(self._tr("rock.loadbc.region.group", "区域载荷 / Region Load"))
            self._btn_add_body_force.setText(self._tr("rock.loadbc.add_body", "添加区域体力"))
            self._btn_add_gravity.setText(self._tr("rock.loadbc.add_gravity", "添加自重"))
            self._btn_pick_load_point.setText(self._tr("rock.loadbc.pick_point_load", "画布选点"))
            self._btn_pick_bc_point.setText(self._tr("rock.loadbc.pick_point_bc", "画布选点"))
            self._btn_pick_bc_edge.setText(self._tr("rock.loadbc.pick_edge_bc", "画布选边"))
            self._btn_bc_fixed.setText(self._tr("rock.loadbc.bc.fixed", "固定 UX/UY"))
            self._btn_bc_fix_x.setText(self._tr("rock.loadbc.bc.fix_x", "约束 UX"))
            self._btn_bc_fix_y.setText(self._tr("rock.loadbc.bc.fix_y", "约束 UY"))
            self._btn_bc_roller_x.setText(self._tr("rock.loadbc.bc.roller_x", "滚动 X"))
            self._btn_bc_roller_y.setText(self._tr("rock.loadbc.bc.roller_y", "滚动 Y"))
            self._bc_target_combo.setItemText(0, self._tr("rock.loadbc.bc.target.point", "点"))
            self._bc_target_combo.setItemText(1, self._tr("rock.loadbc.bc.target.edge", "边"))
            self._bc_target_combo.setItemText(2, self._tr("rock.loadbc.bc.target.all_nodes", "全部节点(旧流程)"))
            self._bc_dir_combo.setItemText(0, self._tr("rock.loadbc.bc.dof.x", "UX"))
            self._bc_dir_combo.setItemText(1, self._tr("rock.loadbc.bc.dof.y", "UY"))
            self._bc_dir_combo.setItemText(2, self._tr("rock.loadbc.bc.dof.xy", "UX/UY"))
            self._bc_dir_combo.setItemText(3, self._tr("rock.loadbc.bc.dof.rot", "旋转(梁/壳预留)"))
            self._btn_apply_bc.setText(self._tr("rock.loadbc.apply_bc", "应用约束"))
            self._load_bc_manager_group.setTitle(self._tr("rock.loadbc.manager.group", "载荷/约束管理器"))
            self._btn_load_manager_apply_edit.setText(self._tr("rock.loadbc.manager.edit", "用当前参数修改载荷"))
            self._btn_load_manager_toggle.setText(self._tr("rock.loadbc.manager.toggle", "激活/抑制"))
            self._btn_load_manager_locate.setText(self._tr("rock.loadbc.manager.locate", "定位"))
            self._btn_load_manager_delete.setText(self._tr("rock.loadbc.manager.delete", "删除"))
            self._btn_bc_manager_apply_edit.setText(self._tr("rock.loadbc.manager.edit_bc", "用当前参数修改约束"))
            self._btn_bc_manager_toggle.setText(self._tr("rock.loadbc.manager.toggle_bc", "激活/抑制"))
            self._btn_bc_manager_locate.setText(self._tr("rock.loadbc.manager.locate_bc", "定位"))
            self._btn_bc_manager_delete.setText(self._tr("rock.loadbc.manager.delete_bc", "删除"))
            self._on_load_bc_target_mode_changed(self._load_bc_target_mode_combo.currentIndex())
            self._on_load_bc_target_set_changed(self._load_bc_target_set_combo.currentIndex())
            self._refresh_load_bc_managers()

            self._rock_mesh_intro.setText(
                self._tr(
                    "rock.mesh.intro",
                    "网格支持全局/局部种子，算法可选 Structured T3、Delaunay T3，以及 GitHub Ear-Clipping 草图三角剖分。",
                )
            )
            self._mesh_backend_label.setText(self._tr("rock.mesh.backend", "Backend"))
            self._mesh_backend_combo.setItemText(0, self._tr("rock.mesh.backend.gmsh", "Gmsh T3"))
            self._mesh_backend_combo.setItemText(1, self._tr("rock.mesh.backend.builtin", "Builtin T3 (Fallback)"))
            self._mesh_element_type_hint.setText(
                self._tr(
                    "rock.mesh.element_type_hint",
                    "当前可求解单元类型: T3。Q4/T6 仅预留扩展接口，暂不生成到求解流程。",
                )
            )
            self._btn_mesh_pick_region.setText(self._tr("rock.mesh.pick_region", "选择区域"))
            self._btn_mesh_apply_region_seed.setText(self._tr("rock.mesh.apply_region_seed", "应用区域种子"))
            self._btn_mesh_pick_edge.setText(self._tr("rock.mesh.pick_edge", "选择边"))
            self._btn_mesh_apply_edge_seed.setText(self._tr("rock.mesh.apply_edge_seed", "应用边种子"))
            self._btn_mesh_generate_selection.setText(self._tr("rock.mesh.generate_selected", "生成选区网格"))
            self._btn_mesh_remesh_selection.setText(self._tr("rock.mesh.remesh_selected", "重划选区网格"))
            self._btn_mesh_delete_selection.setText(self._tr("rock.mesh.delete_selected", "删除选区网格"))
            self._btn_mesh_quality_check.setText(self._tr("rock.mesh.quality.check", "检查质量"))
            if self._mesh_bad_elements_visible:
                self._btn_mesh_toggle_bad.setText(self._tr("rock.mesh.quality.hide_bad", "隐藏坏单元"))
            else:
                self._btn_mesh_toggle_bad.setText(self._tr("rock.mesh.quality.show_bad", "高亮坏单元"))
            self._btn_generate_mesh.setText(self._tr("rock.mesh.generate", "生成网格"))
            self._btn_import_mesh.setText(self._tr("rock.mesh.import", "导入外部网格"))
            self._refresh_mesh_quality_summary()

            self._rock_job_intro.setText(
                self._tr(
                    "rock.job.intro",
                    "设置作业名后提交求解，随后进入可视化查看位移与应力云图。",
                )
            )
            self._btn_submit_job.setText(self._tr("rock.job.submit", "提交作业"))
            self._btn_job_export_canvas.setText(self._tr("rock.job.export_view", "导出视图图像"))

            self._rock_report_intro.setText(
                self._tr(
                    "rock.report.intro",
                    "报告功能支持规范上下文 + AI 工程语境联动，自动汇总位移/应力指标。",
                )
            )
            self._btn_generate_report.setText(self._tr("rock.report.generate", "生成报告"))
            self._btn_export_report.setText(self._tr("rock.report.export", "导出报告"))

            self._geotech_info.setText(
                self._tr(
                    "geotech.info",
                    "Build layered geotechnical model automatically from template text/CSV, then solve with current backend.",
                )
            )
            self._btn_geotech_quick.setText(self._tr("geotech.action.quick", "Quick Geotech Template"))
            self._btn_geotech_text.setText(self._tr("geotech.action.text", "Geotech from Text..."))
            self._btn_geotech_csv.setText(self._tr("geotech.action.csv", "Geotech from CSV..."))
            self._btn_geotech_rerun.setText(self._tr("geotech.action.rerun", "Rerun Last Template"))
            self._btn_geotech_open_output.setText(self._tr("geotech.action.open_output", "Open Output Folder"))
            self._geotech_mesh_size_label.setText(self._tr("geotech.mesh_size", "Mesh Size"))
            self._geotech_top_load_label.setText(self._tr("geotech.top_load", "Top Line Load (Q)"))
            self._geotech_fix_bottom_ux.setText(self._tr("geotech.bc.bottom_ux", "Fix Bottom ux"))
            self._geotech_fix_bottom_uy.setText(self._tr("geotech.bc.bottom_uy", "Fix Bottom uy"))
            self._geotech_fix_lateral_ux.setText(self._tr("geotech.bc.lateral_ux", "Fix Left/Right ux"))
            self._geotech_export_artifacts.setText(self._tr("geotech.export", "Export CSV/SVG artifacts"))
            if self._geotech_point_report_text:
                self._geotech_point_summary.setText(self._geotech_point_report_text)
            else:
                self._geotech_point_summary.setText(self._tr("geotech.points.empty", "No geotechnical point report yet."))

            self._editor_group.setTitle(self._tr("group.editor", "Preprocess Editor (MVP)"))
            self._editor_info.setText(
                self._tr(
                    "editor.info",
                    "Edit nodes/elements/materials directly, validate, then apply to current model.",
                )
            )
            self._editor_canvas_edit_enable.setText(self._tr("editor.canvas.enable", "Enable Canvas Edit"))
            self._editor_canvas_tool_label.setText(self._tr("editor.canvas.tool", "Tool"))
            current_tool = self._selected_canvas_tool()
            self._editor_canvas_tool_combo.blockSignals(True)
            self._editor_canvas_tool_combo.clear()
            self._editor_canvas_tool_combo.addItem(self._tr("editor.canvas.tool.select", "Select/Move"), "select")
            self._editor_canvas_tool_combo.addItem(self._tr("editor.canvas.tool.add_node", "Add Node"), "add_node")
            self._editor_canvas_tool_combo.addItem(self._tr("editor.canvas.tool.add_element", "Add Element"), "add_element")
            self._editor_canvas_tool_combo.addItem(self._tr("editor.canvas.tool.measure", "Measure"), "measure")
            self._editor_canvas_tool_combo.addItem(self._tr("editor.canvas.tool.measure_angle", "Measure Angle"), "measure_angle")
            self._editor_canvas_tool_combo.addItem(self._tr("editor.canvas.tool.pick_geometry", "Pick Geometry"), "pick_geometry")
            idx = self._editor_canvas_tool_combo.findData(current_tool)
            self._editor_canvas_tool_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self._editor_canvas_tool_combo.blockSignals(False)
            self._editor_canvas_material_label.setText(self._tr("editor.canvas.material", "New Element Material"))
            self._btn_editor_load.setText(self._tr("editor.action.load", "Load Current Model"))
            self._btn_editor_validate.setText(self._tr("editor.action.validate", "Validate Draft"))
            self._btn_editor_apply.setText(self._tr("editor.action.apply", "Apply Draft"))
            self._btn_editor_add_material.setText(self._tr("editor.action.add_material", "Add Material Row"))
            self._btn_editor_del_material.setText(self._tr("editor.action.del_material", "Delete Material Row"))
            self._btn_editor_add_node.setText(self._tr("editor.action.add_node", "Add Node Row"))
            self._btn_editor_del_node.setText(self._tr("editor.action.del_node", "Delete Node Row"))
            self._btn_editor_add_element.setText(self._tr("editor.action.add_element", "Add Element Row"))
            self._btn_editor_del_element.setText(self._tr("editor.action.del_element", "Delete Element Row"))

            self._show_deformed.setText(self._tr("view.show_deformed", "Show Deformed"))
            self._show_constraints.setText(self._tr("view.show_constraints", "Show Constraints"))
            self._show_loads.setText(self._tr("view.show_loads", "Show Loads"))
            self._show_node_ids.setText(self._tr("view.show_node_ids", "Show Node IDs"))
            self._show_element_ids.setText(self._tr("view.show_element_ids", "Show Element IDs"))
            self._show_contour.setText(self._tr("view.show_contour", "Show Contour Fill"))
            self._contour_label.setText(self._tr("label.contour_variable", "Contour Variable"))
            self._scale_row_label.setText(self._tr("label.deformation_scale", "Deformation Scale"))
            if self._result is None:
                self._vis_metrics_label.setText(
                    self._tr(
                        "rock.viz.metrics.empty",
                        "No solved result yet. Submit a job to view max displacement/stress metrics.",
                    )
                )

            self._results_tabs.setTabText(0, self._tr("summary.tab", "Summary"))
            self._results_tabs.setTabText(1, self._tr("results.node_tab", "Node Displacements"))
            self._results_tabs.setTabText(2, self._tr("results.element_tab", "Element Strain/Stress"))
            self._results_scope_label.setText(self._tr("results.scope.label", "显示过滤"))
            self._btn_results_scope_clear.setText(self._tr("results.scope.clear", "清除过滤"))
            self._results_scope_type_combo.setItemText(0, self._tr("results.scope.all", "All"))
            self._results_scope_type_combo.setItemText(1, self._tr("results.scope.component", "Component"))
            self._results_scope_type_combo.setItemText(2, self._tr("results.scope.region", "Region"))
            self._results_scope_type_combo.setItemText(3, self._tr("results.scope.material", "Material"))
            self._results_scope_type_combo.setItemText(4, self._tr("results.scope.set", "Set"))
            self._node_filter_label.setText(self._tr("label.filter", "Filter"))
            self._element_filter_label.setText(self._tr("label.filter", "Filter"))
            self._node_filter_edit.setPlaceholderText(self._tr("placeholder.node_filter", "node id or value"))
            self._element_filter_edit.setPlaceholderText(self._tr("placeholder.element_filter", "element id or strain/stress value"))
            self._btn_export_nodes.setText(self._tr("button.export_csv", "Export CSV"))
            self._btn_export_elements.setText(self._tr("button.export_csv", "Export CSV"))
            self._btn_apply_mapping.setText(self._tr("button.apply_mapping", "Apply Mapping Templates"))
            self._btn_log_toggle_tab.setText(self._tr("button.show_hide_log", "Show/Hide Bottom Log Panel"))
            self._btn_log_clear.setText(self._tr("button.clear_log", "Clear Log"))
            self._log_dock.setWindowTitle(self._tr("toolbar.log", "Log"))
            self._log_info.setText(
                self._tr(
                    "log.info",
                    "底部命令历史会记录关键操作，便于回溯当前建模步骤。",
                )
            )
            self._log_hint.setText(
                self._ui("命令历史面板已显示。" if self._command_history.isVisible() else "命令历史面板已隐藏。", "Command history panel is visible." if self._command_history.isVisible() else "Command history panel is hidden.")
            )
            if not self._command_history.toPlainText().strip():
                self._command_history.setPlainText(self._tr("log.history.ready", "RockFEM command history is ready."))

            self._node_table.setHorizontalHeaderLabels(["Node", "ux", "uy"])
            self._element_table.setHorizontalHeaderLabels(["Element", "ex", "ey", "gxy", "sx", "sy", "txy"])
            self._mapping_table.setHorizontalHeaderLabels(["Dim", "Tag", "Name", "Mapped To", "Status", "Items", "Note"])
            self._part_sketch_table.setHorizontalHeaderLabels(
                [
                    self._tr("rock.part.table.id", "Pt"),
                    self._tr("rock.part.table.x", "x"),
                    self._tr("rock.part.table.y", "y"),
                ]
            )
            self._material_manager_table.setHorizontalHeaderLabels(
                [
                    self._tr("material.manager.id", "ID"),
                    self._tr("material.manager.name", "Name"),
                    self._tr("material.manager.e", "Elastic E"),
                    self._tr("material.manager.nu", "Poisson nu"),
                    self._tr("material.manager.mode", "Mode"),
                    self._tr("material.manager.used", "Used"),
                ]
            )
            self._material_usage_details.setPlaceholderText(self._tr("material.manager.details", "Material usage details will appear here."))
            self._component_manager_table.setHorizontalHeaderLabels(
                [
                    self._tr("assembly.component.table.id", "ID"),
                    self._tr("assembly.component.table.name", "Name"),
                    self._tr("assembly.component.table.regions", "Regions"),
                    self._tr("assembly.component.table.elements", "Elements"),
                    self._tr("assembly.component.table.visibility", "Visibility"),
                ]
            )
            self._editor_material_table.setHorizontalHeaderLabels(
                [
                    self._tr("editor.material.id", "material_id"),
                    self._tr("editor.material.e", "young_modulus"),
                    self._tr("editor.material.nu", "poisson_ratio"),
                    self._tr("editor.material.plane_stress", "plane_stress"),
                ]
            )
            self._editor_node_table.setHorizontalHeaderLabels(
                [
                    self._tr("editor.node.id", "node_id"),
                    self._tr("editor.node.x", "x"),
                    self._tr("editor.node.y", "y"),
                ]
            )
            self._editor_element_table.setHorizontalHeaderLabels(
                [
                    self._tr("editor.element.id", "element_id"),
                    self._tr("editor.element.n1", "n1"),
                    self._tr("editor.element.n2", "n2"),
                    self._tr("editor.element.n3", "n3"),
                    self._tr("editor.element.material", "material_id"),
                ]
            )

            self._set_results_state_text()
            self._set_runtime_state(self._runtime_state, self._runtime_detail.text())
            self._update_visualization_metrics()
            self._refresh_component_manager()
        finally:
            self._is_retranslating = False

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _log(self, message: str) -> None:
        self._log_text.appendPlainText(message)
        if hasattr(self, "_command_history"):
            self._command_history.appendPlainText(message)


def launch_app(*, language: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(show_startup_dialog=True, language=language)
    window.show()
    return app.exec()
