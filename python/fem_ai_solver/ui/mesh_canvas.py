
from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, degrees, hypot, sin, sqrt

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QBrush, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QWidget

from fem_ai_solver.fem.model import Model
from fem_ai_solver.fem.results import StaticSolveResult
from fem_ai_solver.ui.colormap import Colormap
from fem_ai_solver.ui.i18n import ui_text


@dataclass(slots=True)
class _ScreenTransform:
    min_x: float
    min_y: float
    scale: float
    height: float
    padding: float
    zoom: float
    pan_x: float
    pan_y: float

    def map(self, x: float, y: float) -> QPointF:
        sx = self.padding + (x - self.min_x) * self.scale * self.zoom + self.pan_x
        sy = self.height - self.padding - (y - self.min_y) * self.scale * self.zoom + self.pan_y
        return QPointF(sx, sy)

    def unmap(self, sx: float, sy: float) -> tuple[float, float]:
        x = (sx - self.padding - self.pan_x) / (self.scale * self.zoom) + self.min_x
        y = ((self.height - self.padding) - sy + self.pan_y) / (self.scale * self.zoom) + self.min_y
        return (x, y)


class MeshCanvas(QWidget):
    node_picked = Signal(int)
    element_picked = Signal(int)
    sketch_face_picked = Signal(float, float)
    node_created = Signal(float, float)
    node_moved = Signal(int, float, float)
    element_created = Signal(list)
    measurement_updated = Signal(float)
    measurement_angle_updated = Signal(float)
    sketch_segment_picked = Signal(int, int)
    sketch_angle_picked = Signal(int, int, int)
    sketch_history_segment_picked = Signal(int, int, int)
    sketch_primitive_drawn = Signal(str, float, float, float, float)
    cancel_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._model: Model | None = None
        self._result: StaticSolveResult | None = None

        self._show_node_ids = False
        self._show_element_ids = False
        self._show_deformed = True
        self._show_constraints = True
        self._show_loads = True

        self._show_contour = True
        self._contour_variable = "sx"
        self._language = "zh_CN"

        self._deformation_scale = 1.0
        self._highlighted_node_ids: set[int] = set()
        self._highlighted_element_ids: set[int] = set()
        self._quality_bad_element_ids: set[int] = set()
        self._active_backend = "python"
        self._runtime_state = "Ready"
        self._results_stale = False
        self._mesh_notice_text = ""

        self._last_pick_node_points: dict[int, QPointF] = {}
        self._last_pick_element_centers: dict[int, QPointF] = {}
        self._pick_node_grid: dict[tuple[int, int], list[int]] = {}
        self._pick_element_grid: dict[tuple[int, int], list[int]] = {}
        self._pick_grid_size = 26.0
        self._pick_index_cache_key: tuple[object, ...] | None = None
        self._last_display_mode = "Undeformed"
        self._current_transform: _ScreenTransform | None = None
        self._current_pick_positions: dict[int, tuple[float, float]] = {}
        self._mesh_path_cache: dict[tuple[object, ...], QPainterPath] = {}
        self._view_zoom = 1.0
        self._view_pan_x = 0.0
        self._view_pan_y = 0.0
        self._is_panning = False
        self._pan_last: QPointF | None = None
        self._sketch_plane: tuple[float, float, float] | None = None
        self._sketch_points: list[tuple[float, float]] = []
        self._sketch_history: list[dict[str, object]] = []
        self._sketch_curve_hint: dict[str, object] | None = None
        self._pickable_sketch_faces: list[list[tuple[float, float]]] = []
        self._highlighted_sketch_faces: list[list[tuple[float, float]]] = []
        self._material_sketch_faces: list[tuple[list[tuple[float, float]], int]] = []
        self._component_sketch_faces: list[tuple[list[tuple[float, float]], QColor]] = []
        self._load_bc_geometry_points: list[dict[str, object]] = []
        self._load_bc_geometry_edges: list[dict[str, object]] = []
        self._load_bc_selected_point_ids: set[str] = set()
        self._load_bc_selected_edge_ids: set[str] = set()
        self._load_bc_geometry_load_previews: list[dict[str, object]] = []
        self._load_bc_geometry_bc_previews: list[dict[str, object]] = []
        self._visible_element_ids: set[int] | None = None
        self._sketch_hover: tuple[float, float] | None = None
        self._selected_sketch_segment: tuple[int, int] | None = None
        self._selected_sketch_angle: tuple[int, int, int] | None = None
        self._pending_measure_nodes: list[int] = []
        self._pending_angle_segment: tuple[int, int] | None = None
        self._primitive_start: tuple[float, float] | None = None
        self._primitive_current: tuple[float, float] | None = None
        self._measure_points: list[tuple[float, float]] = []
        self._measure_text_offset = QPointF(18.0, -14.0)
        self._angle_text_offset = QPointF(22.0, -20.0)
        self._measure_text_rect: QRectF | None = None
        self._angle_text_rect: QRectF | None = None
        self._dragging_measure_text = False
        self._dragging_angle_text = False
        self._drag_last: QPointF | None = None

        self._edit_enabled = False
        self._edit_tool = "select"
        self._dragging_node_id: int | None = None
        self._dragging_node_xy: tuple[float, float] | None = None
        self._pending_element_nodes: list[int] = []

        self._colormap = Colormap()
        self.setMinimumSize(700, 480)
        self.setFocusPolicy(Qt.StrongFocus)

    def _invalidate_render_cache(self) -> None:
        self._mesh_path_cache.clear()
        self._pick_index_cache_key = None

    def set_model(self, model: Model | None) -> None:
        self._model = model
        self._highlighted_node_ids.clear()
        self._highlighted_element_ids.clear()
        self._pickable_sketch_faces.clear()
        self._highlighted_sketch_faces.clear()
        self._material_sketch_faces.clear()
        self._component_sketch_faces.clear()
        self._visible_element_ids = None
        self._quality_bad_element_ids.clear()
        self._mesh_notice_text = ""
        self._invalidate_render_cache()
        self.update()

    def set_load_bc_geometry_overlay(
        self,
        points: list[dict[str, object]],
        edges: list[dict[str, object]],
        selected_point_ids: set[str] | list[str],
        selected_edge_ids: set[str] | list[str],
        load_previews: list[dict[str, object]],
        bc_previews: list[dict[str, object]] | None = None,
    ) -> None:
        self._load_bc_geometry_points = list(points)
        self._load_bc_geometry_edges = list(edges)
        self._load_bc_selected_point_ids = {str(item) for item in selected_point_ids}
        self._load_bc_selected_edge_ids = {str(item) for item in selected_edge_ids}
        self._load_bc_geometry_load_previews = list(load_previews)
        self._load_bc_geometry_bc_previews = list(bc_previews or [])
        self.update()

    def set_result(self, result: StaticSolveResult | None) -> None:
        self._result = result
        self._invalidate_render_cache()
        self.update()

    def set_sketch_plane(self, width: float, height: float, grid_step: float) -> None:
        self._sketch_plane = (max(width, 0.1), max(height, 0.1), max(grid_step, 0.05))
        self._view_zoom = 1.0
        self._view_pan_x = 0.0
        self._view_pan_y = 0.0
        self._measure_points.clear()
        self.update()

    def clear_sketch_plane(self) -> None:
        self._sketch_plane = None
        self._sketch_points.clear()
        self._sketch_history.clear()
        self._sketch_curve_hint = None
        self._pickable_sketch_faces.clear()
        self._sketch_hover = None
        self._measure_points.clear()
        self._pending_measure_nodes.clear()
        self._pending_angle_segment = None
        self._primitive_start = None
        self._primitive_current = None
        self._measure_text_rect = None
        self._angle_text_rect = None
        self.update()

    def reset_view(self) -> None:
        self._view_zoom = 1.0
        self._view_pan_x = 0.0
        self._view_pan_y = 0.0
        self._invalidate_render_cache()
        self.update()

    def zoom_in(self) -> None:
        self._apply_zoom_factor(1.12, anchor=QPointF(self.width() * 0.5, self.height() * 0.5))

    def zoom_out(self) -> None:
        self._apply_zoom_factor(1.0 / 1.12, anchor=QPointF(self.width() * 0.5, self.height() * 0.5))

    def _apply_zoom_factor(self, factor: float, *, anchor: QPointF | None = None) -> None:
        if factor <= 0.0:
            return
        old_transform = self._current_transform
        anchor_world: tuple[float, float] | None = None
        if anchor is not None and old_transform is not None:
            anchor_world = old_transform.unmap(float(anchor.x()), float(anchor.y()))
        new_zoom = max(0.2, min(8.0, self._view_zoom * factor))
        if abs(new_zoom - self._view_zoom) <= 1e-9:
            return
        self._view_zoom = new_zoom
        if anchor is not None and anchor_world is not None and old_transform is not None:
            wx, wy = anchor_world
            sx_without_pan = old_transform.padding + (wx - old_transform.min_x) * old_transform.scale * self._view_zoom
            sy_without_pan = old_transform.height - old_transform.padding - (wy - old_transform.min_y) * old_transform.scale * self._view_zoom
            self._view_pan_x = float(anchor.x()) - sx_without_pan
            self._view_pan_y = float(anchor.y()) - sy_without_pan
        self._invalidate_render_cache()
        self.update()

    def set_sketch_points(self, points: list[tuple[float, float]]) -> None:
        self._sketch_points = [(float(x), float(y)) for x, y in points]
        count = len(self._sketch_points)
        self._pending_measure_nodes = [idx for idx in self._pending_measure_nodes if 0 <= idx < count]
        if self._pending_angle_segment is not None:
            i0, i1 = self._pending_angle_segment
            if not (0 <= i0 < count and 0 <= i1 < count):
                self._pending_angle_segment = None
        if self._selected_sketch_segment is not None:
            i0, i1 = self._selected_sketch_segment
            if not (0 <= i0 < count and 0 <= i1 < count):
                self._selected_sketch_segment = None
        if self._selected_sketch_angle is not None:
            ia, ib, ic = self._selected_sketch_angle
            if not (0 <= ia < count and 0 <= ib < count and 0 <= ic < count):
                self._selected_sketch_angle = None
        self.update()

    def set_sketch_history(self, history: list[dict[str, object]] | None) -> None:
        self._sketch_history = []
        if history:
            for item in history:
                points_raw = item.get("points", [])
                points = [(float(x), float(y)) for x, y in points_raw] if isinstance(points_raw, list) else []
                hint = item.get("hint")
                self._sketch_history.append({"points": points, "hint": hint})
        self.update()

    def set_sketch_curve_hint(self, hint: dict[str, object] | None) -> None:
        self._sketch_curve_hint = hint
        self.update()

    def set_highlighted_sketch_faces(self, faces: list[list[tuple[float, float]]] | None) -> None:
        self._highlighted_sketch_faces = []
        if faces:
            for face in faces:
                normalized = [(float(x), float(y)) for x, y in face]
                if len(normalized) >= 3:
                    self._highlighted_sketch_faces.append(normalized)
        self.update()

    def set_pickable_sketch_faces(self, faces: list[list[tuple[float, float]]] | None) -> None:
        self._pickable_sketch_faces = []
        if faces:
            for face in faces:
                normalized = [(float(x), float(y)) for x, y in face]
                if len(normalized) >= 3:
                    self._pickable_sketch_faces.append(normalized)
        self.update()

    def set_material_sketch_faces(self, faces: list[dict[str, object]] | None) -> None:
        self._material_sketch_faces = []
        if faces:
            for item in faces:
                if not isinstance(item, dict):
                    continue
                points_raw = item.get("points", [])
                material_id = int(item.get("material_id", 0))
                if material_id <= 0 or not isinstance(points_raw, list):
                    continue
                points = [(float(x), float(y)) for x, y in points_raw]
                if len(points) < 3:
                    continue
                self._material_sketch_faces.append((points, material_id))
        self.update()

    def set_component_sketch_faces(self, faces: list[dict[str, object]] | None) -> None:
        self._component_sketch_faces = []
        if faces:
            for item in faces:
                if not isinstance(item, dict):
                    continue
                points_raw = item.get("points", [])
                if not isinstance(points_raw, list):
                    continue
                points = [(float(x), float(y)) for x, y in points_raw]
                if len(points) < 3:
                    continue
                color = QColor(str(item.get("color", "#2563eb")))
                if not color.isValid():
                    color = QColor("#2563eb")
                self._component_sketch_faces.append((points, color))
        self.update()

    def set_visible_elements(self, element_ids: set[int] | list[int] | None) -> None:
        if element_ids is None:
            self._visible_element_ids = None
        else:
            self._visible_element_ids = {int(value) for value in element_ids}
        self._invalidate_render_cache()
        self.update()

    def _is_element_visible(self, element_id: int) -> bool:
        return self._visible_element_ids is None or int(element_id) in self._visible_element_ids

    @staticmethod
    def _material_color(material_id: int, *, alpha: int = 90) -> QColor:
        palette = (
            "#3b82f6",
            "#f59e0b",
            "#10b981",
            "#ef4444",
            "#8b5cf6",
            "#06b6d4",
            "#f97316",
            "#84cc16",
            "#e11d48",
            "#6366f1",
        )
        index = abs(int(material_id)) % len(palette)
        color = QColor(palette[index])
        color.setAlpha(max(0, min(255, int(alpha))))
        return color

    def is_sketch_plane_active(self) -> bool:
        return self._sketch_plane is not None

    def set_selected_sketch_segment(self, segment: tuple[int, int] | None) -> None:
        self._pending_angle_segment = None
        self._pending_measure_nodes.clear()
        self._selected_sketch_segment = None if segment is None else (int(segment[0]), int(segment[1]))
        self._selected_sketch_angle = None
        if self._selected_sketch_segment is not None:
            i0, i1 = self._selected_sketch_segment
            if 0 <= i0 < len(self._sketch_points) and 0 <= i1 < len(self._sketch_points):
                self._measure_points = [self._sketch_points[i0], self._sketch_points[i1]]
        self.update()

    def set_selected_sketch_angle(self, angle_points: tuple[int, int, int] | None) -> None:
        self._pending_angle_segment = None
        self._pending_measure_nodes.clear()
        self._selected_sketch_angle = None if angle_points is None else (
            int(angle_points[0]),
            int(angle_points[1]),
            int(angle_points[2]),
        )
        self._selected_sketch_segment = None
        if self._selected_sketch_angle is not None:
            ia, ib, ic = self._selected_sketch_angle
            if 0 <= ia < len(self._sketch_points) and 0 <= ib < len(self._sketch_points) and 0 <= ic < len(self._sketch_points):
                self._measure_points = [self._sketch_points[ia], self._sketch_points[ib], self._sketch_points[ic]]
        self.update()

    def set_language(self, language: str) -> None:
        self._language = language
        self.update()

    def set_show_node_ids(self, enabled: bool) -> None:
        self._show_node_ids = enabled
        self.update()

    def set_show_element_ids(self, enabled: bool) -> None:
        self._show_element_ids = enabled
        self.update()

    def set_show_deformed(self, enabled: bool) -> None:
        self._show_deformed = enabled
        self._invalidate_render_cache()
        self.update()

    def set_show_constraints(self, enabled: bool) -> None:
        self._show_constraints = enabled
        self.update()

    def set_show_loads(self, enabled: bool) -> None:
        self._show_loads = enabled
        self.update()

    def set_show_contour(self, enabled: bool) -> None:
        self._show_contour = enabled
        self.update()

    def set_contour_variable(self, variable: str) -> None:
        self._contour_variable = variable
        self.update()

    def set_deformation_scale(self, scale: float) -> None:
        self._deformation_scale = max(scale, 0.0)
        self._invalidate_render_cache()
        self.update()

    def set_highlighted_nodes(self, node_ids: set[int] | list[int]) -> None:
        self._highlighted_node_ids = set(node_ids)
        self.update()

    def set_highlighted_elements(self, element_ids: set[int] | list[int]) -> None:
        self._highlighted_element_ids = set(element_ids)
        self.update()

    def set_active_backend(self, backend: str) -> None:
        self._active_backend = backend
        self.update()

    def set_runtime_state(self, state: str, *, results_stale: bool = False) -> None:
        self._runtime_state = state
        self._results_stale = results_stale
        self.update()

    def set_quality_bad_elements(self, element_ids: set[int] | list[int]) -> None:
        self._quality_bad_element_ids = {int(item) for item in element_ids}
        self.update()

    def set_edit_enabled(self, enabled: bool) -> None:
        self._edit_enabled = bool(enabled)
        if not self._edit_enabled:
            self._dragging_node_id = None
            self._dragging_node_xy = None
            self._pending_element_nodes.clear()
        self.update()

    def current_edit_tool(self) -> str:
        return str(self._edit_tool)

    def set_edit_tool(self, tool: str) -> None:
        normalized = tool.strip().lower()
        if normalized not in {"select", "add_node", "add_element", "measure", "measure_angle", "draw_rect", "draw_ellipse", "draw_circle", "pick_geometry"}:
            normalized = "select"
        self._edit_tool = normalized
        self._dragging_node_id = None
        self._dragging_node_xy = None
        self._sketch_hover = None
        if self._edit_tool != "add_element":
            self._pending_element_nodes.clear()
        if self._edit_tool not in {"measure", "measure_angle"}:
            self._measure_points.clear()
            self._pending_measure_nodes.clear()
            self._pending_angle_segment = None
            self._selected_sketch_segment = None
            self._selected_sketch_angle = None
            self._measure_text_rect = None
            self._angle_text_rect = None
            self._dragging_measure_text = False
            self._dragging_angle_text = False
            self._drag_last = None
        if self._edit_tool not in {"draw_rect", "draw_ellipse", "draw_circle"}:
            self._primitive_start = None
            self._primitive_current = None
        self.update()

    def clear_pending_element_nodes(self) -> None:
        self._pending_element_nodes.clear()
        self.update()

    def cancel_temporary_interactions(self) -> None:
        """Clear transient canvas interaction state used by editing tools."""
        self._dragging_node_id = None
        self._dragging_node_xy = None
        self._pending_element_nodes.clear()
        self._measure_points.clear()
        self._pending_measure_nodes.clear()
        self._pending_angle_segment = None
        self._primitive_start = None
        self._primitive_current = None
        self._selected_sketch_segment = None
        self._selected_sketch_angle = None
        self._measure_text_rect = None
        self._angle_text_rect = None
        self._dragging_measure_text = False
        self._dragging_angle_text = False
        self._drag_last = None
        self._sketch_hover = None
        self.update()

    def _handle_measure_pick(self, x: float, y: float) -> None:
        point = (float(x), float(y))
        if self._edit_tool == "measure":
            self._selected_sketch_segment = None
            self._selected_sketch_angle = None
            if len(self._measure_points) >= 2:
                self._measure_points = []
            self._measure_points.append(point)
            if len(self._measure_points) == 2:
                dx = self._measure_points[1][0] - self._measure_points[0][0]
                dy = self._measure_points[1][1] - self._measure_points[0][1]
                self.measurement_updated.emit(float(hypot(dx, dy)))
            self.update()
            return

        if self._edit_tool == "measure_angle":
            self._selected_sketch_segment = None
            self._selected_sketch_angle = None
            if len(self._measure_points) >= 3:
                self._measure_points = []
            self._measure_points.append(point)
            if len(self._measure_points) == 3:
                p1, p2, p3 = self._measure_points
                v1x = p1[0] - p2[0]
                v1y = p1[1] - p2[1]
                v2x = p3[0] - p2[0]
                v2y = p3[1] - p2[1]
                n1 = hypot(v1x, v1y)
                n2 = hypot(v2x, v2y)
                if n1 > 1e-12 and n2 > 1e-12:
                    cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
                    self.measurement_angle_updated.emit(float(degrees(acos(cos_theta))))
            self.update()

    def _pick_nearest_node(self, click: QPointF) -> tuple[int | None, float]:
        best_node_id: int | None = None
        best_node_dist2 = float("inf")
        candidate_ids: list[int] = []
        for key in self._grid_nearby_keys(click):
            candidate_ids.extend(self._pick_node_grid.get(key, []))
        search_ids = candidate_ids if candidate_ids else list(self._last_pick_node_points.keys())
        for node_id in search_ids:
            point = self._last_pick_node_points.get(node_id)
            if point is None:
                continue
            dx = point.x() - click.x()
            dy = point.y() - click.y()
            dist2 = dx * dx + dy * dy
            if dist2 < best_node_dist2:
                best_node_dist2 = dist2
                best_node_id = node_id
        return best_node_id, best_node_dist2

    def _pick_nearest_sketch_point(self, click: QPointF, transform: _ScreenTransform) -> int | None:
        if not self._sketch_points:
            return None
        best_idx: int | None = None
        best_dist2 = float("inf")
        for idx, point in enumerate(self._sketch_points):
            mapped = transform.map(*point)
            dx = mapped.x() - click.x()
            dy = mapped.y() - click.y()
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_idx = idx
        if best_idx is None or best_dist2 > 16.0 * 16.0:
            return None
        return best_idx

    @staticmethod
    def _curve_hint_center(hint: dict[str, object] | None) -> tuple[float, float] | None:
        if not isinstance(hint, dict):
            return None
        kind = str(hint.get("kind", "")).lower()
        bbox = hint.get("bbox")
        if kind not in {"circle", "ellipse"} or not isinstance(bbox, tuple) or len(bbox) != 4:
            return None
        return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)

    def _pick_nearest_measure_point(
        self,
        click: QPointF,
        transform: _ScreenTransform,
    ) -> tuple[tuple[float, float], int | None] | None:
        candidates: list[tuple[tuple[float, float], int | None]] = []
        for idx, point in enumerate(self._sketch_points):
            candidates.append(((float(point[0]), float(point[1])), int(idx)))
        for item in self._sketch_history:
            points = item.get("points")
            if isinstance(points, list):
                for point in points:
                    candidates.append(((float(point[0]), float(point[1])), None))
            center = self._curve_hint_center(item.get("hint") if isinstance(item, dict) else None)
            if center is not None:
                candidates.append((center, None))
        center = self._curve_hint_center(self._sketch_curve_hint)
        if center is not None:
            candidates.append((center, None))
        if not candidates:
            return None
        best: tuple[tuple[float, float], int | None] | None = None
        best_dist2 = float("inf")
        seen: set[tuple[float, float]] = set()
        for point, idx in candidates:
            key = (round(float(point[0]), 9), round(float(point[1]), 9))
            if key in seen:
                continue
            seen.add(key)
            mapped = transform.map(float(point[0]), float(point[1]))
            dx = mapped.x() - click.x()
            dy = mapped.y() - click.y()
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best = (point, idx)
        if best is None or best_dist2 > 16.0 * 16.0:
            return None
        return best

    def _handle_measure_snap_pick(
        self,
        click: QPointF,
        transform: _ScreenTransform,
    ) -> bool:
        picked = self._pick_nearest_measure_point(click, transform)
        if picked is None:
            return False
        point, sketch_idx = picked
        if len(self._measure_points) >= 2:
            self._measure_points = []
            self._pending_measure_nodes.clear()
        if not self._measure_points:
            self._measure_points = [point]
            self._pending_measure_nodes = [int(sketch_idx)] if sketch_idx is not None else []
            self._selected_sketch_segment = None
            self._selected_sketch_angle = None
            self.update()
            return True
        first = self._measure_points[0]
        if hypot(float(point[0]) - float(first[0]), float(point[1]) - float(first[1])) <= 1e-12:
            self.update()
            return True
        first_idx = self._pending_measure_nodes[0] if self._pending_measure_nodes else None
        self._measure_points = [first, point]
        self._pending_measure_nodes.clear()
        if first_idx is not None and sketch_idx is not None and self._is_existing_sketch_edge(int(first_idx), int(sketch_idx)):
            self._selected_sketch_segment = (int(first_idx), int(sketch_idx))
            self.sketch_segment_picked.emit(int(first_idx), int(sketch_idx))
        else:
            self._selected_sketch_segment = None
        self._selected_sketch_angle = None
        dx = self._measure_points[1][0] - self._measure_points[0][0]
        dy = self._measure_points[1][1] - self._measure_points[0][1]
        self.measurement_updated.emit(float(hypot(dx, dy)))
        self.update()
        return True

    def _is_existing_sketch_edge(self, i0: int, i1: int) -> bool:
        if i0 == i1 or len(self._sketch_points) < 2:
            return False
        for idx in range(len(self._sketch_points) - 1):
            a = idx
            b = idx + 1
            if (a == i0 and b == i1) or (a == i1 and b == i0):
                return True
        return False

    @staticmethod
    def _compose_angle_from_segment_order(
        first_seg: tuple[int, int],
        second_seg: tuple[int, int],
    ) -> tuple[int, int, int] | None:
        s1 = {int(first_seg[0]), int(first_seg[1])}
        s2 = {int(second_seg[0]), int(second_seg[1])}
        shared = s1.intersection(s2)
        if len(shared) != 1:
            return None
        vertex = next(iter(shared))
        first_other = int(first_seg[0]) if int(first_seg[1]) == vertex else int(first_seg[1])
        second_other = int(second_seg[0]) if int(second_seg[1]) == vertex else int(second_seg[1])
        if first_other == second_other:
            return None
        return (first_other, vertex, second_other)

    @staticmethod
    def _apply_preview_ratio_lock(
        start: tuple[float, float],
        current: tuple[float, float],
        *,
        ratio: float,
    ) -> tuple[float, float]:
        sx, sy = start
        cx, cy = current
        dx = cx - sx
        dy = cy - sy
        ratio = max(float(ratio), 1e-9)
        adx = abs(dx)
        ady = abs(dy)
        if adx <= 1e-12 and ady <= 1e-12:
            return current
        if adx / ratio > ady:
            ady = adx / ratio
        else:
            adx = ady * ratio
        cx = sx + adx * (1.0 if dx >= 0.0 else -1.0)
        cy = sy + ady * (1.0 if dy >= 0.0 else -1.0)
        return (cx, cy)

    def _current_primitive_preview(self, modifiers: Qt.KeyboardModifier = Qt.NoModifier) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if self._primitive_start is None or self._primitive_current is None:
            return None
        start = self._primitive_start
        current = self._primitive_current
        force_unit = self._edit_tool == "draw_circle" or bool(modifiers & Qt.ShiftModifier)
        if force_unit:
            current = self._apply_preview_ratio_lock(start, current, ratio=1.0)
        return (start, current)

    def _pick_nearest_element(self, click: QPointF) -> tuple[int | None, float]:
        best_element_id: int | None = None
        best_element_dist2 = float("inf")
        candidate_ids: list[int] = []
        for key in self._grid_nearby_keys(click):
            candidate_ids.extend(self._pick_element_grid.get(key, []))
        search_ids = candidate_ids if candidate_ids else list(self._last_pick_element_centers.keys())
        for element_id in search_ids:
            point = self._last_pick_element_centers.get(element_id)
            if point is None:
                continue
            dx = point.x() - click.x()
            dy = point.y() - click.y()
            dist2 = dx * dx + dy * dy
            if dist2 < best_element_dist2:
                best_element_dist2 = dist2
                best_element_id = element_id
        return best_element_id, best_element_dist2

    @staticmethod
    def _point_to_segment_dist2(click: QPointF, p1: QPointF, p2: QPointF) -> float:
        vx = p2.x() - p1.x()
        vy = p2.y() - p1.y()
        wx = click.x() - p1.x()
        wy = click.y() - p1.y()
        vv = vx * vx + vy * vy
        if vv <= 1e-12:
            dx = click.x() - p1.x()
            dy = click.y() - p1.y()
            return dx * dx + dy * dy
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        px = p1.x() + t * vx
        py = p1.y() + t * vy
        dx = click.x() - px
        dy = click.y() - py
        return dx * dx + dy * dy

    def _pick_sketch_segment(self, click: QPointF, transform: _ScreenTransform) -> tuple[int, int] | None:
        if len(self._sketch_points) < 2:
            return None
        best_pair: tuple[int, int] | None = None
        best_dist2 = float("inf")
        for idx in range(len(self._sketch_points) - 1):
            p1 = transform.map(*self._sketch_points[idx])
            p2 = transform.map(*self._sketch_points[idx + 1])
            dist2 = self._point_to_segment_dist2(click, p1, p2)
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_pair = (idx, idx + 1)
        if best_pair is None:
            return None
        if best_dist2 > 11.0 * 11.0:
            return None
        return best_pair

    def _pick_history_segment(self, click: QPointF, transform: _ScreenTransform) -> tuple[int, int, int] | None:
        if not self._sketch_history:
            return None
        best: tuple[int, int, int] | None = None
        best_dist2 = float("inf")
        for h_idx, item in enumerate(self._sketch_history):
            points = item.get("points")
            if not isinstance(points, list) or len(points) < 2:
                continue
            for idx in range(len(points) - 1):
                p1 = transform.map(*points[idx])
                p2 = transform.map(*points[idx + 1])
                dist2 = self._point_to_segment_dist2(click, p1, p2)
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best = (int(h_idx), int(idx), int(idx + 1))
        if best is None or best_dist2 > 11.0 * 11.0:
            return None
        return best

    def _pick_sketch_angle(self, click: QPointF, transform: _ScreenTransform) -> tuple[int, int, int] | None:
        if len(self._sketch_points) < 3:
            return None
        closed = (
            len(self._sketch_points) >= 4
            and abs(self._sketch_points[0][0] - self._sketch_points[-1][0]) <= 1e-9
            and abs(self._sketch_points[0][1] - self._sketch_points[-1][1]) <= 1e-9
        )
        candidates: list[tuple[int, int, int]] = []
        if closed:
            n = len(self._sketch_points) - 1
            for i in range(n):
                candidates.append(((i - 1) % n, i, (i + 1) % n))
        else:
            for i in range(1, len(self._sketch_points) - 1):
                candidates.append((i - 1, i, i + 1))

        best: tuple[int, int, int] | None = None
        best_dist2 = float("inf")
        for triple in candidates:
            prev_pt = transform.map(*self._sketch_points[triple[0]])
            vertex = transform.map(*self._sketch_points[triple[1]])
            next_pt = transform.map(*self._sketch_points[triple[2]])
            dx = vertex.x() - click.x()
            dy = vertex.y() - click.y()
            # Prefer picking actual sketched geometry (vertex or either arm), not arbitrary nearby points.
            dist2_vertex = dx * dx + dy * dy
            dist2_arm1 = self._point_to_segment_dist2(click, vertex, prev_pt)
            dist2_arm2 = self._point_to_segment_dist2(click, vertex, next_pt)
            dist2 = min(dist2_vertex, dist2_arm1, dist2_arm2)
            if dist2 < best_dist2:
                best_dist2 = dist2
                best = triple
        if best is None:
            return None
        if best_dist2 > 12.0 * 12.0:
            return None
        return best

    def _node_position_maps(self) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]] | None]:
        if self._model is None:
            return {}, None

        undeformed = {node.id: (node.x, node.y) for node in self._model.mesh.nodes}

        if self._result is None:
            return undeformed, None

        displacement = self._result.displacements
        deformed: dict[int, tuple[float, float]] = {}
        for index, node in enumerate(self._model.mesh.nodes):
            ux = float(displacement[2 * index])
            uy = float(displacement[2 * index + 1])
            deformed[node.id] = (
                node.x + ux * self._deformation_scale,
                node.y + uy * self._deformation_scale,
            )

        return undeformed, deformed

    def _make_transform(self, points: list[tuple[float, float]]) -> _ScreenTransform:
        if not points:
            return _ScreenTransform(
                0.0,
                0.0,
                1.0,
                float(self.height()),
                26.0,
                self._view_zoom,
                self._view_pan_x,
                self._view_pan_y,
            )

        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)

        width = max(max_x - min_x, 1e-12)
        height = max(max_y - min_y, 1e-12)

        padding = 26.0
        sx = (self.width() - 2.0 * padding) / width
        sy = (self.height() - 2.0 * padding) / height
        scale = min(sx, sy)

        return _ScreenTransform(
            min_x=min_x,
            min_y=min_y,
            scale=scale,
            height=float(self.height()),
            padding=padding,
            zoom=self._view_zoom,
            pan_x=self._view_pan_x,
            pan_y=self._view_pan_y,
        )

    def _element_scalar_field(self) -> tuple[dict[int, float], float, float] | None:
        if self._result is None or self._model is None:
            return None

        values: dict[int, float] = {}

        if self._contour_variable in {"ex", "ey", "gxy", "sx", "sy", "txy"}:
            component_index = {
                "ex": ("strain", 0),
                "ey": ("strain", 1),
                "gxy": ("strain", 2),
                "sx": ("stress", 0),
                "sy": ("stress", 1),
                "txy": ("stress", 2),
            }
            vector_name, index = component_index[self._contour_variable]
            for entry in self._result.element_results:
                vector = entry.strain if vector_name == "strain" else entry.stress
                values[entry.element_id] = float(vector[index])

        elif self._contour_variable == "u_mag":
            displacement = self._result.displacements
            nodal_u: dict[int, float] = {}
            for idx, node in enumerate(self._model.mesh.nodes):
                ux = float(displacement[2 * idx])
                uy = float(displacement[2 * idx + 1])
                nodal_u[node.id] = sqrt(ux * ux + uy * uy)
            for element in self._model.mesh.elements:
                if not element.connectivity:
                    continue
                magnitude = sum(nodal_u.get(node_id, 0.0) for node_id in element.connectivity) / len(element.connectivity)
                values[element.id] = magnitude

        elif self._contour_variable == "s1":
            for entry in self._result.element_results:
                sx, sy, txy = (float(item) for item in entry.stress)
                radius = sqrt(((sx - sy) * 0.5) ** 2 + txy * txy)
                values[entry.element_id] = 0.5 * (sx + sy) + radius

        elif self._contour_variable == "mise":
            for entry in self._result.element_results:
                sx, sy, txy = (float(item) for item in entry.stress)
                values[entry.element_id] = sqrt(max(sx * sx - sx * sy + sy * sy + 3.0 * txy * txy, 0.0))

        elif self._contour_variable == "load_mag":
            nodal_force: dict[int, tuple[float, float]] = {}
            for load in self._model.loads:
                fx, fy = nodal_force.get(load.node_id, (0.0, 0.0))
                if load.dof == "fx":
                    fx += float(load.value)
                else:
                    fy += float(load.value)
                nodal_force[load.node_id] = (fx, fy)
            nodal_mag = {
                node_id: sqrt(fx * fx + fy * fy)
                for node_id, (fx, fy) in nodal_force.items()
            }
            for element in self._model.mesh.elements:
                if not element.connectivity:
                    continue
                magnitude = sum(nodal_mag.get(node_id, 0.0) for node_id in element.connectivity) / len(element.connectivity)
                values[element.id] = magnitude

        else:
            return None

        if not values:
            return None

        vmin = min(values.values())
        vmax = max(values.values())
        if abs(vmax - vmin) <= 1e-14:
            pad = max(abs(vmax), 1.0) * 0.01
            vmin -= pad
            vmax += pad

        return values, vmin, vmax

    def _visible_signature(self) -> tuple[str, int, int]:
        if self._visible_element_ids is None:
            return ("all", 0, 0)
        return ("subset", len(self._visible_element_ids), sum(int(item) for item in self._visible_element_ids))

    def _render_interaction_active(self) -> bool:
        return bool(
            self._is_panning
            or self._dragging_node_id is not None
            or self._primitive_start is not None
            or self._dragging_measure_text
            or self._dragging_angle_text
        )

    def _mesh_path(
        self,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
        *,
        cache_token: str,
        downsample: int = 1,
    ) -> QPainterPath:
        key = (
            cache_token,
            id(self._model),
            id(self._result),
            len(self._model.mesh.elements) if self._model is not None else 0,
            self._visible_signature(),
            round(self._deformation_scale, 6),
            round(transform.min_x, 6),
            round(transform.min_y, 6),
            round(transform.scale, 6),
            round(transform.zoom, 6),
            round(transform.pan_x, 3),
            round(transform.pan_y, 3),
            round(transform.height, 3),
            int(max(downsample, 1)),
        )
        cached = self._mesh_path_cache.get(key)
        if cached is not None:
            return cached
        path = QPainterPath()
        if self._model is not None:
            step = max(int(downsample), 1)
            for idx, element in enumerate(self._model.mesh.elements):
                if step > 1 and idx % step != 0:
                    continue
                if not self._is_element_visible(element.id):
                    continue
                node_ids = element.connectivity
                if len(node_ids) != 3 or any(node_id not in node_positions for node_id in node_ids):
                    continue
                p0 = transform.map(*node_positions[node_ids[0]])
                p1 = transform.map(*node_positions[node_ids[1]])
                p2 = transform.map(*node_positions[node_ids[2]])
                path.moveTo(p0)
                path.lineTo(p1)
                path.lineTo(p2)
                path.lineTo(p0)
        self._mesh_path_cache[key] = path
        return path

    def _grid_key(self, point: QPointF) -> tuple[int, int]:
        size = max(self._pick_grid_size, 1.0)
        return (int(point.x() // size), int(point.y() // size))

    def _grid_nearby_keys(self, point: QPointF) -> list[tuple[int, int]]:
        gx, gy = self._grid_key(point)
        return [(gx + dx, gy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]

    def _draw_contour_fill(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
        scalar_field: dict[int, float],
        vmin: float,
        vmax: float,
    ) -> None:
        if self._model is None:
            return

        painter.setPen(Qt.NoPen)
        for element in self._model.mesh.elements:
            if not self._is_element_visible(element.id):
                continue
            if element.id not in scalar_field:
                continue
            if len(element.connectivity) != 3:
                continue

            node_ids = element.connectivity
            if any(node_id not in node_positions for node_id in node_ids):
                continue

            points = [transform.map(*node_positions[node_id]) for node_id in node_ids]
            polygon = QPolygonF(points)
            value = scalar_field[element.id]
            color = self._colormap.color(value, vmin, vmax)
            painter.setBrush(QBrush(color))
            painter.drawPolygon(polygon)

    def _draw_mesh(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
        *,
        line_color: QColor,
        line_width: float,
        cache_token: str = "mesh",
    ) -> None:
        if self._model is None:
            return

        painter.setPen(QPen(line_color, line_width))
        total_elements = len(self._model.mesh.elements)
        downsample = 1
        if self._render_interaction_active() and total_elements > 20_000:
            downsample = 3 if total_elements > 45_000 else 2
        painter.drawPath(self._mesh_path(transform, node_positions, cache_token=cache_token, downsample=downsample))

    def _draw_material_partition_fill(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
    ) -> None:
        if self._model is None:
            return
        painter.setPen(Qt.NoPen)
        for element in self._model.mesh.elements:
            if not self._is_element_visible(element.id):
                continue
            node_ids = element.connectivity
            if len(node_ids) != 3:
                continue
            if any(node_id not in node_positions for node_id in node_ids):
                continue
            polygon = QPolygonF([transform.map(*node_positions[node_id]) for node_id in node_ids])
            if polygon.count() < 3:
                continue
            fill = self._material_color(int(element.material_id), alpha=68)
            painter.setBrush(QBrush(fill))
            painter.drawPolygon(polygon)

    def _draw_background_grid(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        step: float,
    ) -> None:
        painter.setPen(QPen(QColor("#d5deeb"), 1.0))
        x = xmin
        while x <= xmax + 1e-9:
            p1 = transform.map(x, ymin)
            p2 = transform.map(x, ymax)
            painter.drawLine(p1, p2)
            x += step

        y = ymin
        while y <= ymax + 1e-9:
            p1 = transform.map(xmin, y)
            p2 = transform.map(xmax, y)
            painter.drawLine(p1, p2)
            y += step

    def _draw_axes(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> None:
        origin_x = 0.0 if xmin <= 0.0 <= xmax else xmin
        origin_y = 0.0 if ymin <= 0.0 <= ymax else ymin
        dx = max(xmax - xmin, 1.0)
        dy = max(ymax - ymin, 1.0)
        x_axis_end = min(xmax, origin_x + 0.18 * dx)
        y_axis_end = min(ymax, origin_y + 0.18 * dy)
        origin = transform.map(origin_x, origin_y)
        px = transform.map(x_axis_end, origin_y)
        py = transform.map(origin_x, y_axis_end)

        painter.setPen(QPen(QColor("#b91c1c"), 2.0))
        painter.drawLine(origin, px)
        painter.drawLine(px, QPointF(px.x() - 7.0, px.y() - 3.0))
        painter.drawLine(px, QPointF(px.x() - 7.0, px.y() + 3.0))
        painter.drawText(px + QPointF(8.0, 0.0), "X")

        painter.setPen(QPen(QColor("#1d4ed8"), 2.0))
        painter.drawLine(origin, py)
        painter.drawLine(py, QPointF(py.x() - 3.0, py.y() + 7.0))
        painter.drawLine(py, QPointF(py.x() + 3.0, py.y() + 7.0))
        painter.drawText(py + QPointF(6.0, -6.0), "Y")

    def _draw_sketch_polyline(self, painter: QPainter, transform: _ScreenTransform) -> None:
        self._draw_sketch_history(painter, transform)
        if self._component_sketch_faces:
            for points, color in self._component_sketch_faces:
                polygon = QPolygonF([transform.map(*point) for point in points])
                if polygon.count() < 3:
                    continue
                fill = QColor(color)
                fill.setAlpha(34)
                edge = QColor(color)
                edge.setAlpha(125)
                painter.setPen(QPen(edge, 1.1, Qt.DotLine))
                painter.setBrush(QBrush(fill))
                painter.drawPolygon(polygon)
        if self._material_sketch_faces:
            for points, material_id in self._material_sketch_faces:
                polygon = QPolygonF([transform.map(*point) for point in points])
                if polygon.count() < 3:
                    continue
                fill = self._material_color(material_id, alpha=72)
                edge = self._material_color(material_id, alpha=165)
                painter.setPen(QPen(edge, 1.5))
                painter.setBrush(QBrush(fill))
                painter.drawPolygon(polygon)
        if self._pickable_sketch_faces:
            painter.setPen(QPen(QColor(14, 116, 144, 190), 1.4, Qt.DashLine))
            painter.setBrush(QColor(34, 197, 94, 22))
            for face in self._pickable_sketch_faces:
                polygon = QPolygonF([transform.map(*point) for point in face])
                if polygon.count() >= 3:
                    painter.drawPolygon(polygon)
        if self._highlighted_sketch_faces:
            painter.setPen(QPen(QColor("#f59e0b"), 2.0, Qt.DashLine))
            painter.setBrush(QColor(245, 158, 11, 52))
            for face in self._highlighted_sketch_faces:
                polygon = QPolygonF([transform.map(*point) for point in face])
                if polygon.count() >= 3:
                    painter.drawPolygon(polygon)
        if not self._sketch_points:
            self._draw_primitive_preview(painter, transform)
            return
        if self._sketch_curve_hint is not None and str(self._sketch_curve_hint.get("kind", "")).lower() in {"circle", "ellipse"}:
            self._draw_curve_hint_geometry(painter, transform, self._sketch_curve_hint, outline=QColor("#0f172a"), center_color=QColor("#dc2626"))
        else:
            painter.setPen(QPen(QColor("#0f172a"), 2.0))
            for idx in range(len(self._sketch_points) - 1):
                p1 = transform.map(*self._sketch_points[idx])
                p2 = transform.map(*self._sketch_points[idx + 1])
                painter.drawLine(p1, p2)

        if self._selected_sketch_segment is not None:
            i0, i1 = self._selected_sketch_segment
            if 0 <= i0 < len(self._sketch_points) and 0 <= i1 < len(self._sketch_points):
                painter.setPen(QPen(QColor("#f97316"), 3.0))
                painter.drawLine(
                    transform.map(*self._sketch_points[i0]),
                    transform.map(*self._sketch_points[i1]),
                )
        if self._pending_angle_segment is not None:
            i0, i1 = self._pending_angle_segment
            if 0 <= i0 < len(self._sketch_points) and 0 <= i1 < len(self._sketch_points):
                painter.setPen(QPen(QColor("#0ea5e9"), 2.2, Qt.DashLine))
                painter.drawLine(
                    transform.map(*self._sketch_points[i0]),
                    transform.map(*self._sketch_points[i1]),
                )
        if self._selected_sketch_angle is not None:
            ia, ib, ic = self._selected_sketch_angle
            if 0 <= ia < len(self._sketch_points) and 0 <= ib < len(self._sketch_points) and 0 <= ic < len(self._sketch_points):
                pa = transform.map(*self._sketch_points[ia])
                pb = transform.map(*self._sketch_points[ib])
                pc = transform.map(*self._sketch_points[ic])
                painter.setPen(QPen(QColor("#fb7185"), 3.0))
                painter.drawLine(pb, pa)
                painter.drawLine(pb, pc)
                painter.setBrush(QColor("#fb7185"))
                painter.drawEllipse(pb, 4.4, 4.4)
        if self._pending_measure_nodes:
            painter.setPen(QPen(QColor("#0ea5e9"), 1.4))
            painter.setBrush(QColor("#38bdf8"))
            for idx in self._pending_measure_nodes:
                if 0 <= idx < len(self._sketch_points):
                    painter.drawEllipse(transform.map(*self._sketch_points[idx]), 5.0, 5.0)

        if self._sketch_curve_hint is None:
            painter.setBrush(QColor("#eab308"))
            painter.setPen(QPen(QColor("#a16207"), 1.4))
            for point in self._sketch_points:
                painter.drawEllipse(transform.map(*point), 4.0, 4.0)

        if self._edit_enabled and self._edit_tool == "add_node" and self._sketch_hover is not None:
            painter.setPen(QPen(QColor("#0ea5e9"), 1.8, Qt.DashLine))
            last_point = transform.map(*self._sketch_points[-1])
            hover_point = transform.map(*self._sketch_hover)
            painter.drawLine(last_point, hover_point)
        self._draw_primitive_preview(painter, transform)

    def _draw_primitive_preview(self, painter: QPainter, transform: _ScreenTransform) -> None:
        preview = self._current_primitive_preview(QApplication.keyboardModifiers())
        if preview is None or self._edit_tool not in {"draw_rect", "draw_ellipse", "draw_circle"}:
            return
        (x0, y0), (x1, y1) = preview
        p0 = transform.map(x0, y0)
        p1 = transform.map(x1, y1)
        left = min(p0.x(), p1.x())
        right = max(p0.x(), p1.x())
        top = min(p0.y(), p1.y())
        bottom = max(p0.y(), p1.y())
        if abs(right - left) <= 1.0 or abs(bottom - top) <= 1.0:
            return
        rect = QRectF(left, top, right - left, bottom - top)
        painter.setPen(QPen(QColor("#0ea5e9"), 1.8, Qt.DashLine))
        painter.setBrush(QColor(14, 165, 233, 24))
        if self._edit_tool == "draw_rect":
            painter.drawRect(rect)
        else:
            painter.drawEllipse(rect)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        text = f"W={w:.4g}, H={h:.4g}"
        metrics = painter.fontMetrics()
        tw = float(metrics.horizontalAdvance(text))
        th = float(metrics.height())
        txt_x = rect.right() + 8.0
        txt_y = rect.top() + th + 3.0
        text_rect = QRectF(txt_x - 4.0, txt_y - th, tw + 8.0, th + 6.0)
        painter.fillRect(text_rect, QColor(255, 255, 255, 225))
        painter.setPen(QPen(QColor("#0369a1"), 1.0))
        painter.drawRect(text_rect)
        painter.drawText(QPointF(txt_x, txt_y), text)

    def _draw_sketch_history(self, painter: QPainter, transform: _ScreenTransform) -> None:
        if not self._sketch_history:
            return
        for item in self._sketch_history:
            points = item.get("points")
            hint = item.get("hint")
            if isinstance(hint, dict) and str(hint.get("kind", "")).lower() in {"circle", "ellipse"}:
                self._draw_curve_hint_geometry(painter, transform, hint, outline=QColor("#0f172a"), center_color=QColor("#dc2626"))
                continue
            if not isinstance(points, list) or len(points) < 2:
                continue
            painter.setPen(QPen(QColor("#0f172a"), 2.0))
            for idx in range(len(points) - 1):
                p1 = transform.map(*points[idx])
                p2 = transform.map(*points[idx + 1])
                painter.drawLine(p1, p2)

    def _draw_curve_hint_geometry(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        hint: dict[str, object],
        *,
        outline: QColor,
        center_color: QColor,
    ) -> None:
        bbox = hint.get("bbox")
        center = hint.get("center")
        if not (isinstance(bbox, tuple) and len(bbox) == 4):
            return
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        p0 = transform.map(x0, y0)
        p1 = transform.map(x1, y1)
        left = min(p0.x(), p1.x())
        right = max(p0.x(), p1.x())
        top = min(p0.y(), p1.y())
        bottom = max(p0.y(), p1.y())
        rect = QRectF(left, top, right - left, bottom - top)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(outline, 2.0))
        painter.drawEllipse(rect)
        if isinstance(center, tuple) and len(center) == 2:
            cx, cy = float(center[0]), float(center[1])
            cp = transform.map(cx, cy)
            painter.setPen(QPen(center_color, 1.4))
            painter.drawLine(QPointF(cp.x() - 6.0, cp.y()), QPointF(cp.x() + 6.0, cp.y()))
            painter.drawLine(QPointF(cp.x(), cp.y() - 6.0), QPointF(cp.x(), cp.y() + 6.0))
            painter.drawText(cp + QPointF(7.0, -6.0), "C")

    def _draw_measurement_overlay(self, painter: QPainter, transform: _ScreenTransform) -> None:
        if len(self._measure_points) < 1:
            self._measure_text_rect = None
            self._angle_text_rect = None
            return
        painter.setPen(QPen(QColor("#9333ea"), 2.0, Qt.DashLine))
        mapped_points = [transform.map(*point) for point in self._measure_points]
        for point in mapped_points:
            painter.drawEllipse(point, 3.5, 3.5)

        if self._edit_tool == "measure_angle":
            self._measure_text_rect = None
            if len(mapped_points) >= 2:
                painter.drawLine(mapped_points[0], mapped_points[1])
            if len(mapped_points) < 3:
                self._angle_text_rect = None
                return

            p1, vertex, p3 = mapped_points[0], mapped_points[1], mapped_points[2]
            painter.drawLine(vertex, p3)

            v1x = p1.x() - vertex.x()
            v1y = p1.y() - vertex.y()
            v2x = p3.x() - vertex.x()
            v2y = p3.y() - vertex.y()
            n1 = hypot(v1x, v1y)
            n2 = hypot(v2x, v2y)
            if n1 <= 1e-9 or n2 <= 1e-9:
                self._angle_text_rect = None
                return

            cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
            angle_deg = degrees(acos(cos_theta))

            radius = min(28.0, 0.35 * min(n1, n2))
            p1n = QPointF(vertex.x() + radius * (v1x / n1), vertex.y() + radius * (v1y / n1))
            p2n = QPointF(vertex.x() + radius * (v2x / n2), vertex.y() + radius * (v2y / n2))
            painter.setPen(QPen(QColor("#ec4899"), 1.4))
            painter.drawLine(vertex, p1n)
            painter.drawLine(vertex, p2n)
            self._draw_angle_arc(painter, vertex, (v1x, v1y), (v2x, v2y), radius)

            text = f"{angle_deg:.2f}°"
            metrics = painter.fontMetrics()
            tw = float(metrics.horizontalAdvance(text))
            th = float(metrics.height())
            text_pos = vertex + self._angle_text_offset
            text_rect = QRectF(text_pos.x() - 4.0, text_pos.y() - th, tw + 8.0, th + 6.0)
            painter.fillRect(text_rect, QColor(255, 255, 255, 220))
            painter.setPen(QPen(QColor("#9d174d"), 1.0))
            painter.drawRect(text_rect)
            painter.drawText(QPointF(text_pos.x(), text_pos.y()), text)
            self._angle_text_rect = text_rect
            return

        if len(mapped_points) < 2:
            self._measure_text_rect = None
            self._angle_text_rect = None
            return
        p1, p2 = mapped_points[0], mapped_points[1]
        painter.drawLine(p1, p2)
        self._draw_arrow_head(painter, p1, p2, size=6.0)
        self._draw_arrow_head(painter, p2, p1, size=6.0)
        dx = self._measure_points[1][0] - self._measure_points[0][0]
        dy = self._measure_points[1][1] - self._measure_points[0][1]
        dist = hypot(dx, dy)
        mid = QPointF((p1.x() + p2.x()) * 0.5, (p1.y() + p2.y()) * 0.5)

        text = f"L={dist:.4f}"
        metrics = painter.fontMetrics()
        tw = float(metrics.horizontalAdvance(text))
        th = float(metrics.height())
        text_pos = mid + self._measure_text_offset
        text_rect = QRectF(text_pos.x() - 4.0, text_pos.y() - th, tw + 8.0, th + 6.0)
        painter.fillRect(text_rect, QColor(255, 255, 255, 220))
        painter.setPen(QPen(QColor("#6b21a8"), 1.0))
        painter.drawRect(text_rect)
        painter.drawText(QPointF(text_pos.x(), text_pos.y()), text)
        self._measure_text_rect = text_rect
        self._angle_text_rect = None

    @staticmethod
    def _draw_arrow_head(painter: QPainter, tip: QPointF, other: QPointF, *, size: float = 7.0) -> None:
        vx = tip.x() - other.x()
        vy = tip.y() - other.y()
        norm = hypot(vx, vy)
        if norm <= 1e-9:
            return
        ux = vx / norm
        uy = vy / norm
        px = -uy
        py = ux
        p1 = QPointF(tip.x() - size * ux + 0.45 * size * px, tip.y() - size * uy + 0.45 * size * py)
        p2 = QPointF(tip.x() - size * ux - 0.45 * size * px, tip.y() - size * uy - 0.45 * size * py)
        painter.drawLine(tip, p1)
        painter.drawLine(tip, p2)

    @staticmethod
    def _draw_angle_arc(painter: QPainter, center: QPointF, v1: tuple[float, float], v2: tuple[float, float], radius: float) -> None:
        a1 = atan2(v1[1], v1[0])
        a2 = atan2(v2[1], v2[0])
        delta = a2 - a1
        while delta > 3.141592653589793:
            delta -= 2.0 * 3.141592653589793
        while delta < -3.141592653589793:
            delta += 2.0 * 3.141592653589793
        steps = 28
        points: list[QPointF] = []
        for i in range(steps + 1):
            t = i / steps
            angle = a1 + delta * t
            points.append(QPointF(center.x() + radius * cos(angle), center.y() + radius * sin(angle)))
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def _draw_highlighted_elements(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
    ) -> None:
        if self._model is None or not self._highlighted_element_ids:
            return

        painter.setPen(QPen(QColor("#f57c00"), 2.8))
        painter.setBrush(Qt.NoBrush)
        for element in self._model.mesh.elements:
            if not self._is_element_visible(element.id):
                continue
            if element.id not in self._highlighted_element_ids:
                continue
            node_ids = element.connectivity
            if len(node_ids) != 3:
                continue
            points = [transform.map(*node_positions[node_id]) for node_id in node_ids]
            painter.drawLine(points[0], points[1])
            painter.drawLine(points[1], points[2])
            painter.drawLine(points[2], points[0])

    def _draw_quality_bad_elements(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
    ) -> None:
        if self._model is None or not self._quality_bad_element_ids:
            return
        painter.setPen(QPen(QColor("#dc2626"), 2.0))
        painter.setBrush(Qt.NoBrush)
        for element in self._model.mesh.elements:
            if int(element.id) not in self._quality_bad_element_ids or not self._is_element_visible(element.id):
                continue
            node_ids = element.connectivity
            if len(node_ids) != 3 or any(node_id not in node_positions for node_id in node_ids):
                continue
            points = [transform.map(*node_positions[node_id]) for node_id in node_ids]
            painter.drawLine(points[0], points[1])
            painter.drawLine(points[1], points[2])
            painter.drawLine(points[2], points[0])

    def _draw_highlighted_nodes(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
    ) -> None:
        if not self._highlighted_node_ids:
            return

        painter.setPen(QPen(QColor("#ef6c00"), 1.3))
        painter.setBrush(QColor("#ffcc80"))
        radius = 5.0
        for node_id in self._highlighted_node_ids:
            if node_id not in node_positions:
                continue
            point = transform.map(*node_positions[node_id])
            painter.drawEllipse(point, radius, radius)

    def _draw_ids(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
    ) -> None:
        if self._model is None:
            return

        node_count = len(self._model.mesh.nodes)
        element_count = len(self._model.mesh.elements)
        if (self._show_node_ids and node_count > 3_000) or (self._show_element_ids and element_count > 2_500):
            self._mesh_notice_text = ui_text(
                "canvas.notice.id_hidden",
                self._language,
                "Node/element IDs auto-hidden for large mesh.",
            )
        else:
            self._mesh_notice_text = ""

        if self._show_node_ids:
            if node_count <= 3_000:
                painter.setPen(QPen(QColor("#2f2f2f"), 1.0))
                for node in self._model.mesh.nodes:
                    point = transform.map(*node_positions[node.id])
                    painter.drawText(point + QPointF(4.0, -4.0), str(node.id))

        if self._show_element_ids:
            if element_count <= 2_500:
                painter.setPen(QPen(QColor("#245f74"), 1.0))
                for element in self._model.mesh.elements:
                    if not self._is_element_visible(element.id):
                        continue
                    n1, n2, n3 = element.connectivity
                    x = (node_positions[n1][0] + node_positions[n2][0] + node_positions[n3][0]) / 3.0
                    y = (node_positions[n1][1] + node_positions[n2][1] + node_positions[n3][1]) / 3.0
                    point = transform.map(x, y)
                    painter.drawText(point, f"E{element.id}")

    def _draw_constraints(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
    ) -> None:
        if self._model is None or not self._show_constraints:
            return

        constrained_node_ids = {bc.node_id for bc in self._model.boundary_conditions}
        if not constrained_node_ids:
            return

        painter.setPen(QPen(QColor("#c62828"), 1.2))
        for node_id in constrained_node_ids:
            if node_id not in node_positions:
                continue
            point = transform.map(*node_positions[node_id])
            size = 7.0
            painter.drawRect(point.x() - size / 2.0, point.y() - size / 2.0, size, size)

    def _draw_arrow(self, painter: QPainter, start: QPointF, end: QPointF, color: QColor) -> None:
        painter.setPen(QPen(color, 1.6))
        painter.drawLine(start, end)

        vx = end.x() - start.x()
        vy = end.y() - start.y()
        length = hypot(vx, vy)
        if length <= 1e-9:
            return

        ux = vx / length
        uy = vy / length
        head = 6.2

        left = QPointF(end.x() - head * (ux + 0.5 * uy), end.y() - head * (uy - 0.5 * ux))
        right = QPointF(end.x() - head * (ux - 0.5 * uy), end.y() - head * (uy + 0.5 * ux))
        painter.drawLine(end, left)
        painter.drawLine(end, right)

    def _draw_loads(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
    ) -> None:
        if self._model is None or not self._show_loads:
            return

        loads_by_node: dict[int, tuple[float, float]] = {}
        for load in self._model.loads:
            fx, fy = loads_by_node.get(load.node_id, (0.0, 0.0))
            if load.dof == "fx":
                fx += load.value
            elif load.dof == "fy":
                fy += load.value
            loads_by_node[load.node_id] = (fx, fy)

        if not loads_by_node:
            return

        max_mag = max(hypot(fx, fy) for fx, fy in loads_by_node.values())
        if max_mag <= 1e-14:
            return

        color = QColor("#2e7d32")
        for node_id, (fx, fy) in loads_by_node.items():
            if node_id not in node_positions:
                continue
            magnitude = hypot(fx, fy)
            if magnitude <= 1e-14:
                continue

            base = transform.map(*node_positions[node_id])
            length = 10.0 + 18.0 * (magnitude / max_mag)
            dx = (fx / magnitude) * length
            dy = -(fy / magnitude) * length
            tip = QPointF(base.x() + dx, base.y() + dy)
            self._draw_arrow(painter, base, tip, color)

    def _draw_load_bc_geometry_overlay(self, painter: QPainter, transform: _ScreenTransform) -> None:
        if not (
            self._load_bc_geometry_points
            or self._load_bc_geometry_edges
            or self._load_bc_geometry_load_previews
            or self._load_bc_geometry_bc_previews
        ):
            return

        edge_pen = QPen(QColor("#0f766e"), 1.6, Qt.DashLine)
        selected_edge_pen = QPen(QColor("#f97316"), 3.0)
        painter.setBrush(Qt.NoBrush)
        for item in self._load_bc_geometry_edges:
            edge_id = str(item.get("id", ""))
            try:
                start = transform.map(float(item["x0"]), float(item["y0"]))
                end = transform.map(float(item["x1"]), float(item["y1"]))
            except (KeyError, TypeError, ValueError):
                continue
            painter.setPen(selected_edge_pen if edge_id in self._load_bc_selected_edge_ids else edge_pen)
            painter.drawLine(start, end)
            if edge_id in self._load_bc_selected_edge_ids:
                mid = QPointF((start.x() + end.x()) * 0.5, (start.y() + end.y()) * 0.5)
                painter.setPen(QPen(QColor("#9a3412"), 1.0))
                painter.drawText(mid + QPointF(6.0, -6.0), str(item.get("label", "edge")))

        for item in self._load_bc_geometry_points:
            point_id = str(item.get("id", ""))
            try:
                point = transform.map(float(item["x"]), float(item["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            selected = point_id in self._load_bc_selected_point_ids
            painter.setPen(QPen(QColor("#9a3412" if selected else "#0f766e"), 1.4))
            painter.setBrush(QColor("#fed7aa" if selected else "#ccfbf1"))
            radius = 5.5 if selected else 4.2
            painter.drawEllipse(point, radius, radius)
            if selected:
                painter.drawText(point + QPointF(7.0, -7.0), str(item.get("label", "point")))

        preview_color = QColor("#15803d")
        for item in self._load_bc_geometry_load_previews:
            kind = str(item.get("kind", "point"))
            active = bool(item.get("active", True))
            current_color = preview_color if active else QColor("#94a3b8")
            try:
                vx = float(item.get("vx", 0.0))
                vy = float(item.get("vy", 0.0))
            except (TypeError, ValueError):
                continue
            norm = hypot(vx, vy)
            ux = vx / norm if norm > 1e-12 else 0.0
            uy = vy / norm if norm > 1e-12 else -1.0
            if kind == "region":
                points_raw = item.get("points", [])
                if isinstance(points_raw, list) and len(points_raw) >= 3:
                    polygon = QPolygonF([transform.map(float(px), float(py)) for px, py in points_raw])
                    painter.setPen(QPen(current_color, 1.2, Qt.DotLine))
                    painter.setBrush(QColor(current_color.red(), current_color.green(), current_color.blue(), 28))
                    painter.drawPolygon(polygon)
                try:
                    base = transform.map(float(item["x"]), float(item["y"]))
                except (KeyError, TypeError, ValueError):
                    continue
                tip = QPointF(base.x() + ux * 30.0, base.y() - uy * 30.0)
                self._draw_arrow(painter, base, tip, current_color)
                label = str(item.get("label", ""))
                if label:
                    painter.setPen(QPen(current_color, 1.0))
                    painter.drawText(tip + QPointF(5.0, -5.0), label)
                continue
            if kind == "edge":
                try:
                    x0 = float(item["x0"])
                    y0 = float(item["y0"])
                    x1 = float(item["x1"])
                    y1 = float(item["y1"])
                except (KeyError, TypeError, ValueError):
                    continue
                load_type = str(item.get("load_type", "distributed"))
                direction_mode = str(item.get("direction_mode", "vector"))
                for ratio in (0.2, 0.5, 0.8):
                    x = x0 + (x1 - x0) * ratio
                    y = y0 + (y1 - y0) * ratio
                    local_ux = ux
                    local_uy = uy
                    if load_type == "pressure" and direction_mode in {"normal", "reverse_normal"}:
                        dx = x1 - x0
                        dy = y1 - y0
                        length = hypot(dx, dy)
                        if length > 1e-12:
                            sign = -1.0 if direction_mode == "reverse_normal" else 1.0
                            local_ux = sign * dy / length
                            local_uy = sign * -dx / length
                    base = transform.map(x, y)
                    tip = QPointF(base.x() + local_ux * 24.0, base.y() - local_uy * 24.0)
                    self._draw_arrow(painter, base, tip, current_color)
                continue
            if norm <= 1e-12:
                continue
            try:
                base = transform.map(float(item["x"]), float(item["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            tip = QPointF(base.x() + ux * 28.0, base.y() - uy * 28.0)
            self._draw_arrow(painter, base, tip, current_color)
            label = str(item.get("label", ""))
            if label:
                painter.setPen(QPen(current_color, 1.0))
                painter.drawText(tip + QPointF(5.0, -5.0), label)

        bc_color = QColor("#2563eb")
        for item in self._load_bc_geometry_bc_previews:
            active = bool(item.get("active", True))
            current_color = bc_color if active else QColor("#94a3b8")
            direction = str(item.get("direction", "xy")).upper()
            kind = str(item.get("kind", "point"))
            painter.setPen(QPen(current_color, 1.8))
            painter.setBrush(QColor(current_color.red(), current_color.green(), current_color.blue(), 45))
            if kind == "edge":
                try:
                    start = transform.map(float(item["x0"]), float(item["y0"]))
                    end = transform.map(float(item["x1"]), float(item["y1"]))
                except (KeyError, TypeError, ValueError):
                    continue
                painter.drawLine(start, end)
                for ratio in (0.15, 0.35, 0.55, 0.75, 0.95):
                    base = QPointF(start.x() + (end.x() - start.x()) * ratio, start.y() + (end.y() - start.y()) * ratio)
                    tri = QPolygonF([
                        QPointF(base.x(), base.y()),
                        QPointF(base.x() - 5.0, base.y() + 9.0),
                        QPointF(base.x() + 5.0, base.y() + 9.0),
                    ])
                    painter.drawPolygon(tri)
                mid = QPointF((start.x() + end.x()) * 0.5, (start.y() + end.y()) * 0.5)
                painter.drawText(mid + QPointF(6.0, 13.0), direction)
                continue
            try:
                point = transform.map(float(item["x"]), float(item["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            painter.drawRect(QRectF(point.x() - 6.0, point.y() - 6.0, 12.0, 12.0))
            painter.drawLine(QPointF(point.x() - 9.0, point.y() + 8.0), QPointF(point.x() + 9.0, point.y() + 8.0))
            painter.drawText(point + QPointF(8.0, 10.0), direction)

    def _draw_colorbar(self, painter: QPainter, vmin: float, vmax: float) -> None:
        bar_w = 20.0
        bar_h = min(max(180.0, self.height() * 0.45), 300.0)
        bar_x = self.width() - 48.0
        bar_y = self.height() * 0.5 - bar_h * 0.5

        panel = QRectF(bar_x - 26.0, bar_y - 18.0, 74.0, bar_h + 36.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawRoundedRect(panel, 8.0, 8.0)

        segments = 56
        seg_h = bar_h / segments
        for i in range(segments):
            ratio = 1.0 - (i + 0.5) / segments
            value = vmin + ratio * (vmax - vmin)
            color = self._colormap.color(value, vmin, vmax)
            painter.setBrush(color)
            painter.drawRect(QRectF(bar_x, bar_y + i * seg_h, bar_w, seg_h + 0.5))

        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(bar_x, bar_y, bar_w, bar_h))

        painter.drawText(QRectF(bar_x - 16.0, bar_y - 16.0, 56.0, 14.0), Qt.AlignCenter, self._contour_variable)
        painter.drawText(QRectF(bar_x - 20.0, bar_y - 2.0, 64.0, 14.0), Qt.AlignRight, f"{vmax:.2e}")
        painter.drawText(QRectF(bar_x - 20.0, bar_y + bar_h - 8.0, 64.0, 14.0), Qt.AlignRight, f"{vmin:.2e}")

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.setPen(Qt.NoPen)

        status_box = QRectF(14.0, 12.0, 360.0, 104.0)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawRoundedRect(status_box, 8.0, 8.0)
        painter.setPen(QPen(QColor("#3a4452"), 1.0))

        stale_suffix = " (stale)" if self._results_stale else ""
        mode_label = ui_text("canvas.mode", self._language, "Mode")
        scale_label = ui_text("canvas.scale", self._language, "Scale")
        status_label = ui_text("canvas.status", self._language, "Status")
        quality_line = ""
        if self._quality_bad_element_ids:
            quality_line = (
                f"\n{ui_text('canvas.quality', self._language, 'Quality')}: "
                f"{len(self._quality_bad_element_ids)} {ui_text('canvas.quality.bad', self._language, 'bad elements highlighted')}"
            )
        notice_line = f"\n{self._mesh_notice_text}" if self._mesh_notice_text else ""

        painter.drawText(
            status_box.adjusted(10.0, 8.0, -8.0, -8.0),
            Qt.AlignLeft | Qt.AlignTop,
            f"{mode_label}: {self._last_display_mode}\n"
            f"{scale_label}: {self._deformation_scale:.1f}    Backend: {self._active_backend}\n"
            f"{status_label}: {self._runtime_state}{stale_suffix}"
            f"{quality_line}"
            f"{notice_line}",
        )

        if self._edit_enabled:
            tool_text = {
                "select": ui_text("canvas.edit.tool.select", self._language, "Select/Move"),
                "add_node": ui_text("canvas.edit.tool.add_node", self._language, "Add Node"),
                "add_element": ui_text("canvas.edit.tool.add_element", self._language, "Add Element"),
                "measure": ui_text("canvas.edit.tool.measure", self._language, "Measure"),
                "measure_angle": ui_text("canvas.edit.tool.measure_angle", self._language, "Measure Angle"),
                "draw_rect": ui_text("canvas.edit.tool.draw_rect", self._language, "Draw Rectangle"),
                "draw_ellipse": ui_text("canvas.edit.tool.draw_ellipse", self._language, "Draw Ellipse"),
                "draw_circle": ui_text("canvas.edit.tool.draw_circle", self._language, "Draw Circle"),
            }.get(self._edit_tool, self._edit_tool)
            edit_line = (
                f"{ui_text('canvas.edit.mode', self._language, 'Edit')}: ON    "
                f"{ui_text('canvas.edit.tool', self._language, 'Tool')}: {tool_text}"
            )
            if self._edit_tool == "add_element":
                edit_line += (
                    f"    {ui_text('canvas.edit.pending', self._language, 'Pending')}: "
                    f"{len(self._pending_element_nodes)}/3"
                )
            painter.drawText(status_box.adjusted(10.0, 56.0, -8.0, -8.0), Qt.AlignLeft | Qt.AlignTop, edit_line)

        legend_box = QRectF(14.0, self.height() - 150.0, 262.0, 130.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawRoundedRect(legend_box, 8.0, 8.0)

        painter.setPen(QPen(QColor("#374151"), 1.0))
        painter.drawText(
            legend_box.adjusted(10.0, 6.0, -6.0, -6.0),
            Qt.AlignLeft | Qt.AlignTop,
            ui_text("canvas.legend", self._language, "Legend"),
        )

        y = legend_box.top() + 28.0
        if self._show_contour and self._result is not None:
            self._draw_legend_item(
                painter,
                QPointF(24.0, y),
                QColor("#4a90e2"),
                ui_text("canvas.legend.contour", self._language, "Contour fill"),
            )
            y += 18.0

        self._draw_legend_item(painter, QPointF(24.0, y), QColor("#70757a"), ui_text("canvas.legend.undeformed", self._language, "Undeformed mesh"))
        y += 18.0
        self._draw_legend_item(painter, QPointF(24.0, y), QColor("#145ea8"), ui_text("canvas.legend.deformed", self._language, "Deformed mesh"))
        y += 18.0
        self._draw_legend_item(painter, QPointF(24.0, y), QColor("#c62828"), ui_text("canvas.legend.constraints", self._language, "Constraints"))
        y += 18.0
        self._draw_legend_item(painter, QPointF(24.0, y), QColor("#2e7d32"), ui_text("canvas.legend.loads", self._language, "Loads"))
        y += 18.0
        self._draw_legend_item(painter, QPointF(24.0, y), QColor("#f57c00"), ui_text("canvas.legend.selection", self._language, "Selection"))

        if self._edit_enabled and self._edit_tool in {"add_node", "add_element", "measure", "measure_angle", "draw_rect", "draw_ellipse", "draw_circle"}:
            tool_text = {
                "add_node": ui_text("canvas.edit.tool.add_node", self._language, "Add Node"),
                "add_element": ui_text("canvas.edit.tool.add_element", self._language, "Add Element"),
                "measure": ui_text("canvas.edit.tool.measure", self._language, "Measure"),
                "measure_angle": ui_text("canvas.edit.tool.measure_angle", self._language, "Measure Angle"),
                "draw_rect": ui_text("canvas.edit.tool.draw_rect", self._language, "Draw Rectangle"),
                "draw_ellipse": ui_text("canvas.edit.tool.draw_ellipse", self._language, "Draw Ellipse"),
                "draw_circle": ui_text("canvas.edit.tool.draw_circle", self._language, "Draw Circle"),
            }.get(self._edit_tool, self._edit_tool)
            esc_hint = ui_text("canvas.hint.esc", self._language, "Press Esc to exit current tool")
            hint_text = (
                f"当前工具: {tool_text} | 按 Esc 退出"
                if self._language == "zh_CN"
                else f"Tool: {tool_text} | {esc_hint}"
            )
            metrics = painter.fontMetrics()
            box_w = float(metrics.horizontalAdvance(hint_text)) + 24.0
            box_h = 30.0
            box = QRectF(self.width() - box_w - 14.0, self.height() - box_h - 14.0, box_w, box_h)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(15, 23, 42, 196))
            painter.drawRoundedRect(box, 6.0, 6.0)
            painter.setPen(QPen(QColor("#f8fafc"), 1.0))
            painter.drawText(box.adjusted(10.0, 5.0, -8.0, -4.0), Qt.AlignLeft | Qt.AlignVCenter, hint_text)

    def _draw_legend_item(self, painter: QPainter, start: QPointF, color: QColor, label: str) -> None:
        painter.setPen(QPen(color, 2.0))
        painter.drawLine(start, QPointF(start.x() + 20.0, start.y()))
        painter.setPen(QPen(QColor("#3a4452"), 1.0))
        painter.drawText(QPointF(start.x() + 28.0, start.y() + 4.0), label)

    def _draw_edit_handles(
        self,
        painter: QPainter,
        transform: _ScreenTransform,
        node_positions: dict[int, tuple[float, float]],
    ) -> None:
        if not self._edit_enabled:
            return

        if self._edit_tool == "add_element" and len(self._pending_element_nodes) >= 2:
            painter.setPen(QPen(QColor("#f59e0b"), 2.0, Qt.DashLine))
            for i in range(len(self._pending_element_nodes) - 1):
                n1 = self._pending_element_nodes[i]
                n2 = self._pending_element_nodes[i + 1]
                if n1 in node_positions and n2 in node_positions:
                    p1 = transform.map(*node_positions[n1])
                    p2 = transform.map(*node_positions[n2])
                    painter.drawLine(p1, p2)

        if self._pending_element_nodes:
            painter.setPen(QPen(QColor("#ca8a04"), 1.5))
            painter.setBrush(QColor("#facc15"))
            for node_id in self._pending_element_nodes:
                if node_id in node_positions:
                    painter.drawEllipse(transform.map(*node_positions[node_id]), 4.2, 4.2)

        if self._dragging_node_id is not None and self._dragging_node_xy is not None:
            painter.setPen(QPen(QColor("#ea580c"), 1.6))
            painter.setBrush(QColor("#fdba74"))
            point = transform.map(*self._dragging_node_xy)
            painter.drawEllipse(point, 5.5, 5.5)

    def _build_pick_index(
        self,
        transform: _ScreenTransform,
        positions: dict[int, tuple[float, float]],
    ) -> None:
        cache_key = (
            id(self._model),
            id(self._result),
            len(positions),
            self._visible_signature(),
            round(self._deformation_scale, 6),
            round(transform.min_x, 6),
            round(transform.min_y, 6),
            round(transform.scale, 6),
            round(transform.zoom, 6),
            round(transform.pan_x, 3),
            round(transform.pan_y, 3),
            round(transform.height, 3),
        )
        if cache_key == self._pick_index_cache_key:
            return
        self._pick_index_cache_key = cache_key
        self._last_pick_node_points = {
            node_id: transform.map(*coords)
            for node_id, coords in positions.items()
        }
        self._pick_node_grid = {}
        for node_id, point in self._last_pick_node_points.items():
            self._pick_node_grid.setdefault(self._grid_key(point), []).append(int(node_id))

        self._last_pick_element_centers = {}
        self._pick_element_grid = {}
        if self._model is None:
            return

        for element in self._model.mesh.elements:
            if not self._is_element_visible(element.id):
                continue
            if len(element.connectivity) != 3:
                continue
            n1, n2, n3 = element.connectivity
            if n1 not in positions or n2 not in positions or n3 not in positions:
                continue
            cx = (positions[n1][0] + positions[n2][0] + positions[n3][0]) / 3.0
            cy = (positions[n1][1] + positions[n2][1] + positions[n3][1]) / 3.0
            center = transform.map(cx, cy)
            self._last_pick_element_centers[element.id] = center
            self._pick_element_grid.setdefault(self._grid_key(center), []).append(int(element.id))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.setFocus(Qt.MouseFocusReason)
        if event.button() in {Qt.MiddleButton, Qt.RightButton}:
            self._is_panning = True
            self._pan_last = event.position()
            return

        if (
            self._sketch_plane is not None
            and self._edit_enabled
            and event.button() == Qt.LeftButton
            and self._current_transform is not None
            and self._model is not None
            and not self._model.mesh.nodes
        ):
            click = event.position()
            allow_free_measure_pick = bool(event.modifiers() & Qt.ShiftModifier)
            if self._edit_tool == "measure" and self._measure_text_rect is not None and self._measure_text_rect.contains(click):
                self._dragging_measure_text = True
                self._drag_last = click
                return
            if self._edit_tool == "measure_angle" and self._angle_text_rect is not None and self._angle_text_rect.contains(click):
                self._dragging_angle_text = True
                self._drag_last = click
                return
            x, y = self._current_transform.unmap(float(event.position().x()), float(event.position().y()))
            if self._edit_tool == "add_node":
                self.node_created.emit(float(x), float(y))
                return
            if self._edit_tool in {"draw_rect", "draw_ellipse", "draw_circle"}:
                self._primitive_start = (float(x), float(y))
                self._primitive_current = (float(x), float(y))
                self.update()
                return
            if self._edit_tool in {"measure", "measure_angle"}:
                click = event.position()
                if self._edit_tool == "measure":
                    if not allow_free_measure_pick and self._handle_measure_snap_pick(click, self._current_transform):
                        return
                    self._handle_measure_pick(float(x), float(y))
                    return
                if self._edit_tool == "measure_angle":
                    if len(self._sketch_points) >= 3 and not allow_free_measure_pick:
                        segment = self._pick_sketch_segment(click, self._current_transform)
                        if segment is not None:
                            if self._pending_angle_segment is None:
                                self._pending_angle_segment = segment
                                self._selected_sketch_segment = segment
                                self._selected_sketch_angle = None
                                self.update()
                                return
                            triple = self._compose_angle_from_segment_order(self._pending_angle_segment, segment)
                            if triple is None:
                                self._pending_angle_segment = segment
                                self._selected_sketch_segment = segment
                                self._selected_sketch_angle = None
                                self.update()
                                return
                            ia, ib, ic = triple
                            self._pending_angle_segment = None
                            self._selected_sketch_angle = triple
                            self._selected_sketch_segment = None
                            self._measure_points = [self._sketch_points[ia], self._sketch_points[ib], self._sketch_points[ic]]
                            p1, p2, p3 = self._measure_points
                            v1x = p1[0] - p2[0]
                            v1y = p1[1] - p2[1]
                            v2x = p3[0] - p2[0]
                            v2y = p3[1] - p2[1]
                            n1 = hypot(v1x, v1y)
                            n2 = hypot(v2x, v2y)
                            if n1 > 1e-12 and n2 > 1e-12:
                                cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
                                self.measurement_angle_updated.emit(float(degrees(acos(cos_theta))))
                            self.sketch_angle_picked.emit(ia, ib, ic)
                            self.update()
                            return
                        history_segment = self._pick_history_segment(click, self._current_transform)
                        if history_segment is not None:
                            h_idx, i0, i1 = history_segment
                            self.sketch_history_segment_picked.emit(int(h_idx), int(i0), int(i1))
                            self.update()
                            return
                    triple = self._pick_sketch_angle(click, self._current_transform)
                    if triple is not None:
                        ia, ib, ic = triple
                        self._selected_sketch_angle = triple
                        self._selected_sketch_segment = None
                        self._measure_points = [self._sketch_points[ia], self._sketch_points[ib], self._sketch_points[ic]]
                        p1, p2, p3 = self._measure_points
                        v1x = p1[0] - p2[0]
                        v1y = p1[1] - p2[1]
                        v2x = p3[0] - p2[0]
                        v2y = p3[1] - p2[1]
                        n1 = hypot(v1x, v1y)
                        n2 = hypot(v2x, v2y)
                        if n1 > 1e-12 and n2 > 1e-12:
                            cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
                            self.measurement_angle_updated.emit(float(degrees(acos(cos_theta))))
                        self.sketch_angle_picked.emit(ia, ib, ic)
                        self.update()
                        return
                    if len(self._sketch_points) >= 3 and not allow_free_measure_pick:
                        return
                self._handle_measure_pick(float(x), float(y))
                return
            if self._edit_tool == "select":
                self.sketch_face_picked.emit(float(x), float(y))
                return
            if self._edit_tool == "pick_geometry":
                self.sketch_face_picked.emit(float(x), float(y))
                return

        if self._model is None:
            return super().mousePressEvent(event)

        click = event.position()
        if self._edit_enabled and event.button() == Qt.LeftButton:
            transform = self._current_transform
            if transform is None:
                return
            allow_free_measure_pick = bool(event.modifiers() & Qt.ShiftModifier)

            if self._edit_tool in {"measure", "measure_angle"}:
                if self._edit_tool == "measure" and self._measure_text_rect is not None and self._measure_text_rect.contains(click):
                    self._dragging_measure_text = True
                    self._drag_last = click
                    return
                if self._edit_tool == "measure_angle" and self._angle_text_rect is not None and self._angle_text_rect.contains(click):
                    self._dragging_angle_text = True
                    self._drag_last = click
                    return
                if self._edit_tool == "measure":
                    if not allow_free_measure_pick and self._handle_measure_snap_pick(click, transform):
                        return
                    x, y = transform.unmap(float(click.x()), float(click.y()))
                    self._handle_measure_pick(float(x), float(y))
                    return
                if self._edit_tool == "measure_angle":
                    if len(self._sketch_points) >= 3 and not allow_free_measure_pick:
                        segment = self._pick_sketch_segment(click, transform)
                        if segment is not None:
                            if self._pending_angle_segment is None:
                                self._pending_angle_segment = segment
                                self._selected_sketch_segment = segment
                                self._selected_sketch_angle = None
                                self.update()
                                return
                            triple = self._compose_angle_from_segment_order(self._pending_angle_segment, segment)
                            if triple is None:
                                self._pending_angle_segment = segment
                                self._selected_sketch_segment = segment
                                self._selected_sketch_angle = None
                                self.update()
                                return
                            ia, ib, ic = triple
                            self._pending_angle_segment = None
                            self._selected_sketch_angle = triple
                            self._selected_sketch_segment = None
                            self._measure_points = [self._sketch_points[ia], self._sketch_points[ib], self._sketch_points[ic]]
                            p1, p2, p3 = self._measure_points
                            v1x = p1[0] - p2[0]
                            v1y = p1[1] - p2[1]
                            v2x = p3[0] - p2[0]
                            v2y = p3[1] - p2[1]
                            n1 = hypot(v1x, v1y)
                            n2 = hypot(v2x, v2y)
                            if n1 > 1e-12 and n2 > 1e-12:
                                cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
                                self.measurement_angle_updated.emit(float(degrees(acos(cos_theta))))
                            self.sketch_angle_picked.emit(ia, ib, ic)
                            self.update()
                            return
                        history_segment = self._pick_history_segment(click, transform)
                        if history_segment is not None:
                            h_idx, i0, i1 = history_segment
                            self.sketch_history_segment_picked.emit(int(h_idx), int(i0), int(i1))
                            self.update()
                            return
                    triple = self._pick_sketch_angle(click, transform)
                    if triple is not None:
                        ia, ib, ic = triple
                        self._selected_sketch_angle = triple
                        self._selected_sketch_segment = None
                        self._measure_points = [self._sketch_points[ia], self._sketch_points[ib], self._sketch_points[ic]]
                        p1, p2, p3 = self._measure_points
                        v1x = p1[0] - p2[0]
                        v1y = p1[1] - p2[1]
                        v2x = p3[0] - p2[0]
                        v2y = p3[1] - p2[1]
                        n1 = hypot(v1x, v1y)
                        n2 = hypot(v2x, v2y)
                        if n1 > 1e-12 and n2 > 1e-12:
                            cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
                            self.measurement_angle_updated.emit(float(degrees(acos(cos_theta))))
                        self.sketch_angle_picked.emit(ia, ib, ic)
                        self.update()
                        return
                    if len(self._sketch_points) >= 3 and not allow_free_measure_pick:
                        return
                x, y = transform.unmap(float(click.x()), float(click.y()))
                self._handle_measure_pick(float(x), float(y))
                return

            if self._edit_tool == "add_node":
                x, y = transform.unmap(float(click.x()), float(click.y()))
                self.node_created.emit(float(x), float(y))
                return
            if self._edit_tool == "pick_geometry":
                x, y = transform.unmap(float(click.x()), float(click.y()))
                self.sketch_face_picked.emit(float(x), float(y))
                return
            if self._edit_tool in {"draw_rect", "draw_ellipse", "draw_circle"}:
                x, y = transform.unmap(float(click.x()), float(click.y()))
                self._primitive_start = (float(x), float(y))
                self._primitive_current = (float(x), float(y))
                self.update()
                return

            nearest_node_id, nearest_node_dist2 = self._pick_nearest_node(click)
            if self._edit_tool == "add_element":
                if nearest_node_id is not None and nearest_node_dist2 <= 12.0 * 12.0:
                    if nearest_node_id not in self._pending_element_nodes:
                        self._pending_element_nodes.append(nearest_node_id)
                        self.node_picked.emit(nearest_node_id)
                        if len(self._pending_element_nodes) == 3:
                            self.element_created.emit(list(self._pending_element_nodes))
                            self._pending_element_nodes.clear()
                    self.update()
                return

            if self._edit_tool == "select":
                if nearest_node_id is not None and nearest_node_dist2 <= 11.0 * 11.0:
                    self.node_picked.emit(nearest_node_id)
                    self._dragging_node_id = nearest_node_id
                    self._dragging_node_xy = self._current_pick_positions.get(nearest_node_id)
                    self.update()
                    return

                nearest_element_id, nearest_element_dist2 = self._pick_nearest_element(click)
                if nearest_element_id is not None and nearest_element_dist2 <= 18.0 * 18.0:
                    self.element_picked.emit(nearest_element_id)
                    return
                return

        best_node_id, best_node_dist2 = self._pick_nearest_node(click)

        if best_node_id is not None and best_node_dist2 <= 11.0 * 11.0:
            self.node_picked.emit(best_node_id)
            return

        best_element_id, best_element_dist2 = self._pick_nearest_element(click)

        if best_element_id is not None and best_element_dist2 <= 18.0 * 18.0:
            self.element_picked.emit(best_element_id)
            return

        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if (self._dragging_measure_text or self._dragging_angle_text) and self._drag_last is not None:
            delta = event.position() - self._drag_last
            self._drag_last = event.position()
            if self._dragging_measure_text:
                self._measure_text_offset = QPointF(
                    self._measure_text_offset.x() + float(delta.x()),
                    self._measure_text_offset.y() + float(delta.y()),
                )
            if self._dragging_angle_text:
                self._angle_text_offset = QPointF(
                    self._angle_text_offset.x() + float(delta.x()),
                    self._angle_text_offset.y() + float(delta.y()),
                )
            self.update()
            return

        if self._is_panning and self._pan_last is not None:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self._view_pan_x += float(delta.x())
            self._view_pan_y += float(delta.y())
            self._invalidate_render_cache()
            self.update()
            return

        if (
            self._edit_enabled
            and self._edit_tool in {"draw_rect", "draw_ellipse", "draw_circle"}
            and self._primitive_start is not None
            and self._current_transform is not None
        ):
            x, y = self._current_transform.unmap(float(event.position().x()), float(event.position().y()))
            self._primitive_current = (float(x), float(y))
            self.update()
            return

        if (
            self._sketch_plane is not None
            and self._edit_enabled
            and self._edit_tool == "add_node"
            and self._current_transform is not None
            and self._model is not None
            and not self._model.mesh.nodes
        ):
            x, y = self._current_transform.unmap(float(event.position().x()), float(event.position().y()))
            self._sketch_hover = (float(x), float(y))
            self.update()
            return

        if (
            self._edit_enabled
            and self._edit_tool == "select"
            and self._dragging_node_id is not None
            and self._current_transform is not None
        ):
            x, y = self._current_transform.unmap(float(event.position().x()), float(event.position().y()))
            self._dragging_node_xy = (float(x), float(y))
            self.update()
            return
        return super().mouseMoveEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key_Escape:
            self.cancel_temporary_interactions()
            self.cancel_requested.emit()
            event.accept()
            return
        return super().keyPressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton and (self._dragging_measure_text or self._dragging_angle_text):
            self._dragging_measure_text = False
            self._dragging_angle_text = False
            self._drag_last = None
            return

        if event.button() in {Qt.MiddleButton, Qt.RightButton}:
            self._is_panning = False
            self._pan_last = None
            return

        if (
            event.button() == Qt.LeftButton
            and self._edit_enabled
            and self._edit_tool in {"draw_rect", "draw_ellipse", "draw_circle"}
            and self._primitive_start is not None
        ):
            if self._current_transform is not None:
                x, y = self._current_transform.unmap(float(event.position().x()), float(event.position().y()))
                self._primitive_current = (float(x), float(y))
            preview = self._current_primitive_preview(event.modifiers())
            self._primitive_start = None
            self._primitive_current = None
            if preview is not None:
                (x0, y0), (x1, y1) = preview
                if abs(x1 - x0) > 1e-9 and abs(y1 - y0) > 1e-9:
                    kind = "rect" if self._edit_tool == "draw_rect" else "ellipse"
                    if self._edit_tool == "draw_circle":
                        kind = "circle"
                    self.sketch_primitive_drawn.emit(kind, float(x0), float(y0), float(x1), float(y1))
            self.update()
            return

        if (
            self._edit_enabled
            and self._edit_tool == "select"
            and event.button() == Qt.LeftButton
            and self._dragging_node_id is not None
            and self._current_transform is not None
        ):
            x, y = self._current_transform.unmap(float(event.position().x()), float(event.position().y()))
            self.node_moved.emit(int(self._dragging_node_id), float(x), float(y))
            self._dragging_node_id = None
            self._dragging_node_xy = None
            self.update()
            return
        return super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        steps = float(delta) / 120.0
        factor = 1.12 ** steps
        self._apply_zoom_factor(factor, anchor=event.position())
        event.accept()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#eef2f7"))

        if self._model is None or not self._model.mesh.nodes:
            if self._sketch_plane is not None:
                width, height, step = self._sketch_plane
                transform = self._make_transform([(0.0, 0.0), (width, height)])
                self._current_transform = transform
                self._draw_background_grid(
                    painter,
                    transform,
                    xmin=0.0,
                    xmax=width,
                    ymin=0.0,
                    ymax=height,
                    step=step,
                )
                self._draw_axes(painter, transform, 0.0, width, 0.0, height)
                self._draw_sketch_polyline(painter, transform)
                self._draw_load_bc_geometry_overlay(painter, transform)
                self._draw_measurement_overlay(painter, transform)
                painter.setPen(QPen(QColor("#334155"), 1.0))
                painter.drawText(
                    QRectF(18.0, self.height() - 36.0, self.width() - 36.0, 24.0),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    ui_text("canvas.sketch.hint", self._language, "Sketch plane: use Part tools to add points."),
                )
                self._draw_overlay(painter)
                return
            painter.setPen(QPen(QColor("#333333"), 1.0))
            painter.drawText(self.rect(), Qt.AlignCenter, ui_text("canvas.no_model", self._language, "No model loaded"))
            return

        undeformed, deformed = self._node_position_maps()
        all_points = list(undeformed.values())
        if self._show_deformed and deformed is not None:
            all_points.extend(deformed.values())

        transform = self._make_transform(all_points)
        self._current_transform = transform
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        width_extent = max_x - min_x
        height_extent = max_y - min_y
        grid_step = max(min(width_extent, height_extent) / 20.0, 0.5)
        interaction_heavy = self._render_interaction_active()
        element_count = len(self._model.mesh.elements)
        self._draw_background_grid(
            painter,
            transform,
            xmin=min_x,
            xmax=max_x,
            ymin=min_y,
            ymax=max_y,
            step=grid_step,
        )
        self._draw_axes(painter, transform, min_x, max_x, min_y, max_y)

        scalar_bundle = self._element_scalar_field()
        if self._show_contour and scalar_bundle is not None and not (interaction_heavy and element_count > 12_000):
            scalar_field, vmin, vmax = scalar_bundle
            self._draw_contour_fill(painter, transform, undeformed, scalar_field, vmin, vmax)
            self._draw_colorbar(painter, vmin, vmax)
        elif not (interaction_heavy and element_count > 18_000):
            self._draw_material_partition_fill(painter, transform, undeformed)

        self._draw_mesh(painter, transform, undeformed, line_color=QColor("#70757a"), line_width=1.0, cache_token="undeformed")

        pick_positions = undeformed
        if self._show_deformed and deformed is not None and not self._edit_enabled:
            self._draw_mesh(painter, transform, deformed, line_color=QColor("#145ea8"), line_width=1.4, cache_token="deformed")
            pick_positions = deformed
            self._last_display_mode = ui_text("canvas.mode.overlay", self._language, "Undeformed + Deformed")
        else:
            self._last_display_mode = ui_text("canvas.mode.undeformed", self._language, "Undeformed")
            if self._show_deformed and deformed is not None and self._edit_enabled:
                self._draw_mesh(painter, transform, deformed, line_color=QColor("#145ea8"), line_width=1.0, cache_token="deformed")
                self._last_display_mode = ui_text("canvas.mode.overlay", self._language, "Undeformed + Deformed")
            pick_positions = undeformed

        self._build_pick_index(transform, pick_positions)
        self._current_pick_positions = dict(pick_positions)

        self._draw_quality_bad_elements(painter, transform, pick_positions)
        self._draw_highlighted_elements(painter, transform, pick_positions)
        self._draw_constraints(painter, transform, undeformed)
        self._draw_loads(painter, transform, undeformed)
        self._draw_load_bc_geometry_overlay(painter, transform)
        self._draw_edit_handles(painter, transform, undeformed)
        self._draw_measurement_overlay(painter, transform)
        self._draw_highlighted_nodes(painter, transform, pick_positions)
        self._draw_ids(painter, transform, pick_positions)
        self._draw_overlay(painter)
