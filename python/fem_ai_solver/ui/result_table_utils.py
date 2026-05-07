from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from fem_ai_solver.fem.results import StaticSolveResult


def build_node_rows(result: StaticSolveResult) -> list[tuple[int, float, float]]:
    return [
        (item.node_id, item.ux, item.uy)
        for item in result.node_displacements
    ]


def build_element_rows(
    result: StaticSolveResult,
) -> list[tuple[int, float, float, float, float, float, float]]:
    return [
        (
            item.element_id,
            item.strain[0],
            item.strain[1],
            item.strain[2],
            item.stress[0],
            item.stress[1],
            item.stress[2],
        )
        for item in result.element_results
    ]


def filter_row_indices(rows: Sequence[Sequence[object]], query: str) -> list[int]:
    text = query.strip().lower()
    if not text:
        return list(range(len(rows)))

    visible: list[int] = []
    for index, row in enumerate(rows):
        blob = " ".join(str(value).lower() for value in row)
        if text in blob:
            visible.append(index)
    return visible


def export_rows_to_csv(
    path: str | Path,
    headers: Iterable[str],
    rows: Iterable[Sequence[object]],
) -> None:
    file_path = Path(path)
    with file_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow(list(row))
