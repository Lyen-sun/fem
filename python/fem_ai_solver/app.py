import sys

from PySide6.QtWidgets import QApplication

from fem_ai_solver.ui.i18n import install_qt_translator, load_language
from fem_ai_solver.ui.main_window import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    install_qt_translator(app)

    language = load_language()
    window = MainWindow(show_startup_dialog=True, language=language)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
