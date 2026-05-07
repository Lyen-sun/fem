from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor


@dataclass(frozen=True, slots=True)
class UiPalette:
    background: str = "#f3f5f8"
    panel_bg: str = "#ffffff"
    panel_alt_bg: str = "#f8fafc"
    border: str = "#d7dde5"
    text_primary: str = "#1f2937"
    text_secondary: str = "#6b7280"
    accent: str = "#0b66d6"
    accent_soft: str = "#e6f0ff"
    success: str = "#1f9d55"
    warning: str = "#c27a0a"
    danger: str = "#b42318"


PALETTE = UiPalette()


def apply_engineering_light_theme(widget) -> None:
    widget.setStyleSheet(
        f"""
        QMainWindow {{
            background: {PALETTE.background};
            color: {PALETTE.text_primary};
        }}
        QWidget#HeaderWidget {{
            background: {PALETTE.panel_bg};
            border-bottom: 1px solid {PALETTE.border};
        }}
        QLabel#BrandTitle {{
            font-size: 20px;
            font-weight: 700;
            color: {PALETTE.text_primary};
        }}
        QLabel#BrandIcon {{
            border: 1px solid {PALETTE.border};
            border-radius: 8px;
            background: {PALETTE.panel_alt_bg};
            padding: 3px;
        }}
        QWidget#WorkflowStrip QPushButton {{
            background: {PALETTE.panel_alt_bg};
            border: 1px solid {PALETTE.border};
            border-radius: 7px;
            padding: 6px 10px;
            font-weight: 600;
        }}
        QWidget#WorkflowStrip QPushButton:checked {{
            background: {PALETTE.accent};
            color: white;
            border-color: {PALETTE.accent};
        }}
        QWidget#QuickToolStrip QToolButton {{
            min-width: 32px;
            min-height: 28px;
            padding: 4px;
            border-radius: 7px;
            background: {PALETTE.panel_alt_bg};
        }}
        QWidget#PartToolStrip QToolButton {{
            min-width: 32px;
            min-height: 28px;
            padding: 4px;
            border-radius: 7px;
            background: {PALETTE.panel_alt_bg};
        }}
        QWidget#PartToolStrip QComboBox, QWidget#PartToolStrip QDoubleSpinBox {{
            min-height: 28px;
        }}
        QMenuBar {{
            background: {PALETTE.panel_bg};
            border: 1px solid {PALETTE.border};
            border-radius: 8px;
            padding: 2px 4px;
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 6px 10px;
            margin: 2px;
            border-radius: 6px;
        }}
        QMenuBar::item:selected {{
            background: {PALETTE.accent_soft};
        }}
        QToolBar {{
            background: {PALETTE.panel_bg};
            border-bottom: 1px solid {PALETTE.border};
            spacing: 8px;
            padding: 6px 10px;
        }}
        QToolButton {{
            background: {PALETTE.panel_alt_bg};
            border: 1px solid {PALETTE.border};
            border-radius: 8px;
            padding: 6px 10px;
            color: {PALETTE.text_primary};
        }}
        QToolButton:hover {{
            background: {PALETTE.accent_soft};
            border-color: {PALETTE.accent};
        }}
        QToolButton:pressed {{
            background: #dbeafe;
        }}
        QToolButton#PrimaryToolButton {{
            background: {PALETTE.accent};
            color: white;
            border-color: {PALETTE.accent};
            font-weight: 600;
        }}
        QToolButton#PrimaryToolButton:hover {{
            background: #0a58b8;
        }}
        QWidget#SidePanel {{
            background: {PALETTE.panel_bg};
            border: 1px solid {PALETTE.border};
            border-radius: 10px;
        }}
        QTabWidget::pane {{
            border: 1px solid {PALETTE.border};
            border-radius: 10px;
            background: {PALETTE.panel_bg};
        }}
        QTabBar::tab {{
            background: {PALETTE.panel_alt_bg};
            border: 1px solid {PALETTE.border};
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 7px 12px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background: {PALETTE.panel_bg};
            color: {PALETTE.accent};
            font-weight: 600;
        }}
        QGroupBox {{
            background: {PALETTE.panel_bg};
            border: 1px solid {PALETTE.border};
            border-radius: 10px;
            margin-top: 8px;
            padding: 10px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: {PALETTE.text_primary};
        }}
        QLabel#SubtleLabel {{
            color: {PALETTE.text_secondary};
        }}
        QLabel#StatusBadge {{
            border-radius: 8px;
            padding: 3px 8px;
            font-weight: 600;
        }}
        QPushButton {{
            background: {PALETTE.panel_alt_bg};
            border: 1px solid {PALETTE.border};
            border-radius: 8px;
            padding: 6px 10px;
        }}
        QPushButton:hover {{
            border-color: {PALETTE.accent};
            background: {PALETTE.accent_soft};
        }}
        QPushButton#PrimaryButton {{
            background: {PALETTE.accent};
            color: white;
            border-color: {PALETTE.accent};
            font-weight: 600;
        }}
        QPushButton#PrimaryButton:hover {{
            background: #0a58b8;
        }}
        QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget, QTreeWidget {{
            background: {PALETTE.panel_bg};
            border: 1px solid {PALETTE.border};
            border-radius: 8px;
            padding: 4px;
            selection-background-color: #cfe3ff;
            selection-color: {PALETTE.text_primary};
        }}
        QHeaderView::section {{
            background: {PALETTE.panel_alt_bg};
            border: 1px solid {PALETTE.border};
            padding: 5px;
            font-weight: 600;
        }}
        QSplitter::handle {{
            background: {PALETTE.border};
        }}
        """
    )


def mapping_row_color(status: str) -> QColor:
    key = status.lower()
    if key == "mapped":
        return QColor("#ebf8ef")
    if key == "warning":
        return QColor("#fff4e5")
    if key == "unrecognized":
        return QColor("#fdecec")
    return QColor("#ffffff")


def status_badge_style(status: str) -> str:
    key = status.lower()
    if key in ("ready", "solved"):
        bg = "#ebf8ef"
        fg = PALETTE.success
    elif key in ("warning", "stale"):
        bg = "#fff4e5"
        fg = PALETTE.warning
    elif key in ("solving",):
        bg = "#e6f0ff"
        fg = PALETTE.accent
    elif key in ("error",):
        bg = "#fdecec"
        fg = PALETTE.danger
    else:
        bg = "#eef2f7"
        fg = PALETTE.text_secondary
    return f"background:{bg}; color:{fg}; border:1px solid {PALETTE.border};"
