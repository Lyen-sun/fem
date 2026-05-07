"""FEM AI Solver package."""

__all__ = ["main"]


def main() -> int:
    from .app import main as app_main

    return app_main()
