from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fem_ai_solver.fem.model import Element, Mesh, Node


@dataclass(slots=True)
class GmshT3ImportResult:
    mesh: Mesh
    physical_name_by_dim_tag: dict[tuple[int, int], str]
    triangle_physical_tag_by_element_id: dict[int, int]
    line_node_ids_by_physical_tag: dict[int, list[int]]


def _read_text(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"mesh file does not exist: {file_path}")
    return file_path.read_text(encoding="utf-8")


def _find_section(lines: list[str], name: str) -> tuple[int, int] | None:
    start_tag = f"${name}"
    end_tag = f"$End{name}"
    if start_tag not in lines or end_tag not in lines:
        return None

    start = lines.index(start_tag)
    end = lines.index(end_tag)
    if end <= start + 1:
        raise ValueError(f"Gmsh section {name} is empty")
    return start + 1, end


def load_t3_mesh_from_json(path: str | Path) -> Mesh:
    """Load T3 mesh data from a JSON file.

    Expected schema:
    {
      "nodes": [{"id": 1, "x": 0.0, "y": 0.0}, ...],
      "elements": [{"id": 1, "connectivity": [1,2,3], "material_id": 1}, ...]
    }
    """
    payload = json.loads(_read_text(path))
    raw_nodes = payload.get("nodes")
    raw_elements = payload.get("elements")

    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("JSON mesh must include a non-empty 'nodes' array")
    if not isinstance(raw_elements, list) or not raw_elements:
        raise ValueError("JSON mesh must include a non-empty 'elements' array")

    nodes = [
        Node(id=int(item["id"]), x=float(item["x"]), y=float(item["y"]))
        for item in raw_nodes
    ]
    elements = [
        Element(
            id=int(item["id"]),
            type="T3",
            connectivity=[int(value) for value in item["connectivity"]],
            material_id=int(item.get("material_id", 1)),
        )
        for item in raw_elements
    ]

    return Mesh(
        nodes=sorted(nodes, key=lambda value: value.id),
        elements=sorted(elements, key=lambda value: value.id),
    )


def load_t3_mesh_from_gmsh_msh_with_physical_groups(
    path: str | Path,
    *,
    default_material_id: int = 1,
) -> GmshT3ImportResult:
    """Load T3 mesh and physical-group metadata from Gmsh ASCII .msh v2.

    Supported in stage2:
    - `$PhysicalNames` optional section.
    - `$Nodes`, `$Elements` sections.
    - Triangle elements (type 2) are imported.
    - Line elements (type 1) are collected for boundary/load template mapping.

    Not supported yet:
    - Binary .msh files.
    - Gmsh v4 entity-block parsing.
    - High-order elements.
    """
    lines = [line.strip() for line in _read_text(path).splitlines() if line.strip()]

    nodes_section = _find_section(lines, "Nodes")
    elements_section = _find_section(lines, "Elements")
    if nodes_section is None or elements_section is None:
        raise ValueError("Gmsh file must contain both $Nodes and $Elements sections")

    physical_name_by_dim_tag: dict[tuple[int, int], str] = {}
    physical_section = _find_section(lines, "PhysicalNames")
    if physical_section is not None:
        start, end = physical_section
        try:
            count = int(lines[start])
        except ValueError as exc:
            raise ValueError("Invalid Gmsh physical-name count line") from exc

        records = lines[start + 1:end]
        if len(records) != count:
            raise ValueError("Gmsh physical-name count does not match actual records")

        for record in records:
            parts = record.split(maxsplit=2)
            if len(parts) < 3:
                raise ValueError(f"Invalid Gmsh physical-name line: {record}")
            dim = int(parts[0])
            tag = int(parts[1])
            name = parts[2].strip()
            if name.startswith('"') and name.endswith('"') and len(name) >= 2:
                name = name[1:-1]
            physical_name_by_dim_tag[(dim, tag)] = name

    node_start, node_end = nodes_section
    try:
        node_count = int(lines[node_start])
    except ValueError as exc:
        raise ValueError("Invalid Gmsh node count line") from exc

    node_lines = lines[node_start + 1:node_end]
    if len(node_lines) != node_count:
        raise ValueError("Gmsh node section count does not match actual node records")

    nodes: list[Node] = []
    for line in node_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid Gmsh node line: {line}")
        node_id = int(fields[0])
        x = float(fields[1])
        y = float(fields[2])
        nodes.append(Node(id=node_id, x=x, y=y))

    element_start, element_end = elements_section
    try:
        element_count = int(lines[element_start])
    except ValueError as exc:
        raise ValueError("Invalid Gmsh element count line") from exc

    element_lines = lines[element_start + 1:element_end]
    if len(element_lines) != element_count:
        raise ValueError("Gmsh element section count does not match actual element records")

    elements: list[Element] = []
    triangle_physical_tag_by_element_id: dict[int, int] = {}
    line_node_sets_by_physical_tag: dict[int, set[int]] = {}

    for line in element_lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid Gmsh element line: {line}")

        element_id = int(fields[0])
        element_type = int(fields[1])
        tag_count = int(fields[2])
        if len(fields) < 3 + tag_count:
            raise ValueError(f"Invalid Gmsh element tags in line: {line}")

        tags = [int(value) for value in fields[3:3 + tag_count]]
        node_fields = fields[3 + tag_count:]
        physical_tag = tags[0] if tags else None

        if element_type == 2:
            if len(node_fields) < 3:
                raise ValueError(f"Invalid T3 connectivity in line: {line}")
            connectivity = [int(node_fields[0]), int(node_fields[1]), int(node_fields[2])]
            material_id = physical_tag if physical_tag is not None else default_material_id
            elements.append(
                Element(
                    id=element_id,
                    type="T3",
                    connectivity=connectivity,
                    material_id=material_id,
                )
            )
            if physical_tag is not None:
                triangle_physical_tag_by_element_id[element_id] = physical_tag
        elif element_type == 1 and physical_tag is not None:
            if len(node_fields) < 2:
                raise ValueError(f"Invalid line-element connectivity in line: {line}")
            node_set = line_node_sets_by_physical_tag.setdefault(physical_tag, set())
            node_set.add(int(node_fields[0]))
            node_set.add(int(node_fields[1]))

    if not elements:
        raise ValueError("No T3 triangle elements (Gmsh type 2) found in file")

    line_node_ids_by_physical_tag = {
        tag: sorted(node_ids)
        for tag, node_ids in line_node_sets_by_physical_tag.items()
    }

    mesh = Mesh(
        nodes=sorted(nodes, key=lambda value: value.id),
        elements=sorted(elements, key=lambda value: value.id),
    )
    return GmshT3ImportResult(
        mesh=mesh,
        physical_name_by_dim_tag=physical_name_by_dim_tag,
        triangle_physical_tag_by_element_id=triangle_physical_tag_by_element_id,
        line_node_ids_by_physical_tag=line_node_ids_by_physical_tag,
    )


def load_t3_mesh_from_gmsh_msh(path: str | Path, *, material_id: int = 1) -> Mesh:
    """Load T3 triangles from a Gmsh ASCII .msh v2 file.

    This compatibility wrapper preserves the stage1 return type (`Mesh`) while
    stage2 metadata is available through
    `load_t3_mesh_from_gmsh_msh_with_physical_groups`.
    """
    return load_t3_mesh_from_gmsh_msh_with_physical_groups(
        path,
        default_material_id=material_id,
    ).mesh
