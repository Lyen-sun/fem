from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from fem_ai_solver.ui.i18n import ui_text


@dataclass(frozen=True, slots=True)
class StartupChoice:
    action: str


class StartupDialog(QDialog):
    def __init__(self, language: str, parent=None) -> None:
        super().__init__(parent)
        self._language = language
        self.choice = StartupChoice(action="new")

        self.setModal(True)
        self.setMinimumWidth(360)

        self._title_label = QLabel()
        self._subtitle_label = QLabel()

        self._btn_new = QPushButton()
        self._btn_import = QPushButton()
        self._btn_demo = QPushButton()

        self._btn_new.clicked.connect(lambda: self._accept_with("new"))
        self._btn_import.clicked.connect(lambda: self._accept_with("import"))
        self._btn_demo.clicked.connect(lambda: self._accept_with("demo"))

        self._btn_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._title_label)
        layout.addWidget(self._subtitle_label)
        layout.addWidget(self._btn_new)
        layout.addWidget(self._btn_import)
        layout.addWidget(self._btn_demo)
        layout.addWidget(self._btn_box)

        self.retranslate(language)

    def _accept_with(self, action: str) -> None:
        self.choice = StartupChoice(action=action)
        self.accept()

    def retranslate(self, language: str) -> None:
        self._language = language
        self.setWindowTitle(ui_text("startup.title", language, "Start FEM AI Solver"))
        self._title_label.setText(ui_text("startup.title", language, "Start FEM AI Solver"))
        self._subtitle_label.setText(ui_text("startup.subtitle", language, "Choose how to start"))
        self._btn_new.setText(ui_text("startup.new_empty", language, "New Empty Model"))
        self._btn_import.setText(ui_text("startup.import_model", language, "Import Model File"))
        self._btn_demo.setText(ui_text("startup.load_demo", language, "Load Demo Model"))
        cancel_button = self._btn_box.button(QDialogButtonBox.Cancel)
        if cancel_button is not None:
            cancel_button.setText(ui_text("startup.cancel", language, "Cancel"))
