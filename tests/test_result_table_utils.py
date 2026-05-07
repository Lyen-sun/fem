from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fem_ai_solver.fem.results import (
    ElementStrainStress,
    NodeDisplacement,
    StaticSolveResult,
    StaticSolveSummary,
)
from fem_ai_solver.ui.result_table_utils import (
    build_element_rows,
    build_node_rows,
    export_rows_to_csv,
    filter_row_indices,
)

_RUNTIME_DIR = Path("tests/.tmp_runtime")
_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _sample_result() -> StaticSolveResult:
    import numpy as np

    return StaticSolveResult(
        displacements=np.array([0.0, 0.0, 1e-6, -2e-6], dtype=np.float64),
        reactions=np.array([10.0, -10.0, 0.0, 0.0], dtype=np.float64),
        node_displacements=[
            NodeDisplacement(node_id=1, ux=0.0, uy=0.0),
            NodeDisplacement(node_id=2, ux=1e-6, uy=-2e-6),
        ],
        element_results=[
            ElementStrainStress(
                element_id=1,
                strain=(1e-6, 2e-6, 3e-6),
                stress=(100.0, 200.0, 300.0),
            )
        ],
        summary=StaticSolveSummary(
            node_count=2,
            element_count=1,
            total_dof=4,
            max_displacement=2.2360679e-6,
        ),
    )


def test_build_rows_and_filter_indices() -> None:
    result = _sample_result()

    node_rows = build_node_rows(result)
    element_rows = build_element_rows(result)

    assert node_rows[1][0] == 2
    assert element_rows[0][0] == 1

    assert filter_row_indices(node_rows, "") == [0, 1]
    assert filter_row_indices(node_rows, "2") == [1]
    assert filter_row_indices(element_rows, "300.0") == [0]


def test_export_rows_to_csv() -> None:
    path = _RUNTIME_DIR / f"rows_{uuid4().hex}.csv"
    rows = [(1, 0.0, 0.0), (2, 1e-6, -2e-6)]

    export_rows_to_csv(path, ["node_id", "ux", "uy"], rows)
    text = path.read_text(encoding="utf-8")
    path.unlink(missing_ok=True)

    assert "node_id,ux,uy" in text
    assert "1,0.0,0.0" in text
    assert "2,1e-06,-2e-06" in text
