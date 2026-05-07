from __future__ import annotations

from dataclasses import dataclass, replace

from fem_ai_solver.fem.model import BoundaryCondition, Load, Material, Model, Node
from fem_ai_solver.mesh.importers import GmshT3ImportResult


@dataclass(slots=True)
class PhysicalGroupMappingEntry:
    dimension: int
    tag: int
    name: str
    mapped_to: str
    status: str
    note: str
    item_count: int


@dataclass(slots=True)
class TemplateApplicationReport:
    material_count: int
    boundary_condition_count: int
    load_count: int
    recognized_group_count: int
    mapping_entries: list[PhysicalGroupMappingEntry]


def _signed_double_area(n1: Node, n2: Node, n3: Node) -> float:
    return (n2.x - n1.x) * (n3.y - n1.y) - (n3.x - n1.x) * (n2.y - n1.y)


def _parse_group_rule(name: str) -> tuple[str, float | None]:
    text = name.strip()
    if not text:
        return "", None

    if "=" in text:
        key, value_text = text.split("=", 1)
        key = key.strip().upper()
        value = float(value_text.strip())
        return key, value

    return text.upper(), None


def summarize_model_state(model: Model, backend: str = "python") -> dict[str, int | str]:
    constrained_nodes = {bc.node_id for bc in model.boundary_conditions}
    loaded_nodes = {load.node_id for load in model.loads}
    return {
        "node_count": len(model.mesh.nodes),
        "element_count": len(model.mesh.elements),
        "material_count": len(model.materials),
        "boundary_condition_count": len(model.boundary_conditions),
        "load_count": len(model.loads),
        "constrained_node_count": len(constrained_nodes),
        "loaded_node_count": len(loaded_nodes),
        "backend": backend,
    }


def summarize_mapping_entries(
    entries: list[PhysicalGroupMappingEntry],
) -> dict[str, int]:
    mapped = sum(1 for entry in entries if entry.status == "mapped")
    warning = sum(1 for entry in entries if entry.status == "warning")
    unrecognized = sum(1 for entry in entries if entry.status == "unrecognized")
    return {
        "total_group_count": len(entries),
        "mapped_group_count": mapped,
        "warning_group_count": warning,
        "unrecognized_group_count": unrecognized,
    }


def build_physical_group_mapping_preview(
    gmsh_import: GmshT3ImportResult,
) -> list[PhysicalGroupMappingEntry]:
    entries: list[PhysicalGroupMappingEntry] = []

    line_tags = set(gmsh_import.line_node_ids_by_physical_tag.keys())
    tri_tags = set(gmsh_import.triangle_physical_tag_by_element_id.values())
    declared_tags = set(gmsh_import.physical_name_by_dim_tag.keys())

    all_groups = set(declared_tags)
    all_groups.update((1, tag) for tag in line_tags)
    all_groups.update((2, tag) for tag in tri_tags)

    for dim, tag in sorted(all_groups):
        name = gmsh_import.physical_name_by_dim_tag.get((dim, tag), "")
        pretty_name = name if name else "<unnamed>"

        if dim == 2:
            element_count = sum(
                1
                for element_tag in gmsh_import.triangle_physical_tag_by_element_id.values()
                if element_tag == tag
            )
            if element_count == 0:
                entries.append(
                    PhysicalGroupMappingEntry(
                        dimension=dim,
                        tag=tag,
                        name=pretty_name,
                        mapped_to="material partition",
                        status="warning",
                        note="Declared but not used by any T3 element.",
                        item_count=0,
                    )
                )
            else:
                entries.append(
                    PhysicalGroupMappingEntry(
                        dimension=dim,
                        tag=tag,
                        name=pretty_name,
                        mapped_to=f"material_id={tag}",
                        status="mapped",
                        note="Mapped from triangle physical tag to element material id.",
                        item_count=element_count,
                    )
                )
            continue

        if dim != 1:
            entries.append(
                PhysicalGroupMappingEntry(
                    dimension=dim,
                    tag=tag,
                    name=pretty_name,
                    mapped_to="unsupported",
                    status="unrecognized",
                    note="Only line (dim=1) and surface (dim=2) groups are used in stage2.",
                    item_count=0,
                )
            )
            continue

        node_ids = gmsh_import.line_node_ids_by_physical_tag.get(tag, [])
        rule, value = _parse_group_rule(name)

        if not rule:
            entries.append(
                PhysicalGroupMappingEntry(
                    dimension=dim,
                    tag=tag,
                    name=pretty_name,
                    mapped_to="unrecognized",
                    status="unrecognized",
                    note="No naming rule matched.",
                    item_count=len(node_ids),
                )
            )
            continue

        if rule in ("BC_FIX", "BC_FIX_XY"):
            entries.append(
                PhysicalGroupMappingEntry(
                    dimension=dim,
                    tag=tag,
                    name=pretty_name,
                    mapped_to="BC ux=0, uy=0",
                    status="mapped",
                    note="Applied to all nodes in this line group.",
                    item_count=len(node_ids),
                )
            )
        elif rule in ("BC_FIX_X", "BC_UX"):
            prescribed = 0.0 if value is None else value
            status = "mapped"
            note = "Applied ux template to group nodes."
            if rule == "BC_UX" and value is None:
                status = "warning"
                note = "BC_UX has no explicit value, defaulted to 0.0."
            entries.append(
                PhysicalGroupMappingEntry(
                    dimension=dim,
                    tag=tag,
                    name=pretty_name,
                    mapped_to=f"BC ux={prescribed}",
                    status=status,
                    note=note,
                    item_count=len(node_ids),
                )
            )
        elif rule in ("BC_FIX_Y", "BC_UY"):
            prescribed = 0.0 if value is None else value
            status = "mapped"
            note = "Applied uy template to group nodes."
            if rule == "BC_UY" and value is None:
                status = "warning"
                note = "BC_UY has no explicit value, defaulted to 0.0."
            entries.append(
                PhysicalGroupMappingEntry(
                    dimension=dim,
                    tag=tag,
                    name=pretty_name,
                    mapped_to=f"BC uy={prescribed}",
                    status=status,
                    note=note,
                    item_count=len(node_ids),
                )
            )
        elif rule in ("LOAD_FX", "LOAD_FY"):
            direction = "fx" if rule == "LOAD_FX" else "fy"
            total = 0.0 if value is None else value
            status = "mapped"
            note = f"Distributed total {direction} equally to group nodes."
            if value is None:
                status = "warning"
                note = f"{rule} has no explicit value, defaulted to 0.0."
            entries.append(
                PhysicalGroupMappingEntry(
                    dimension=dim,
                    tag=tag,
                    name=pretty_name,
                    mapped_to=f"Load {direction} total={total}",
                    status=status,
                    note=note,
                    item_count=len(node_ids),
                )
            )
        else:
            entries.append(
                PhysicalGroupMappingEntry(
                    dimension=dim,
                    tag=tag,
                    name=pretty_name,
                    mapped_to="unrecognized",
                    status="unrecognized",
                    note="Name does not match supported BC/LOAD rules.",
                    item_count=len(node_ids),
                )
            )

    return entries


def collect_pre_solve_warnings(model: Model) -> list[str]:
    warnings: list[str] = []
    if not model.boundary_conditions:
        warnings.append("No displacement boundary condition found; system may be singular.")
    if not model.loads:
        warnings.append("No nodal loads found; solution may be trivially zero.")
    if len(model.materials) > 1:
        warnings.append("Multiple materials detected; verify material partition mapping.")
    return warnings


def validate_model(model: Model, area_tolerance: float = 1e-14) -> None:
    """Validate model references and basic T3 geometry before solve."""
    if area_tolerance <= 0.0:
        raise ValueError("area_tolerance must be positive")

    if not model.mesh.nodes:
        raise ValueError("model must contain at least one node")
    if not model.mesh.elements:
        raise ValueError("model must contain at least one element")
    if not model.materials:
        raise ValueError("model must contain at least one material")

    node_by_id: dict[int, Node] = {}
    for node in model.mesh.nodes:
        if node.id in node_by_id:
            raise ValueError(f"duplicate node id detected: {node.id}")
        node_by_id[node.id] = node

    material_by_id: dict[int, Material] = {}
    for material in model.materials:
        if material.id in material_by_id:
            raise ValueError(f"duplicate material id detected: {material.id}")
        if material.young_modulus <= 0.0:
            raise ValueError("young_modulus must be positive")
        if material.poisson_ratio <= -1.0 or material.poisson_ratio >= 0.5:
            raise ValueError("poisson_ratio must be in (-1.0, 0.5)")
        material_by_id[material.id] = material

    for element in model.mesh.elements:
        if element.type != "T3":
            raise ValueError(
                f"Element {element.id} has unsupported type '{element.type}'. solve_linear_static currently supports only T3 elements."
            )
        if len(element.connectivity) != 3:
            raise ValueError(
                f"Element {element.id} must have exactly 3 node ids in connectivity for T3."
            )

        try:
            n1 = node_by_id[element.connectivity[0]]
            n2 = node_by_id[element.connectivity[1]]
            n3 = node_by_id[element.connectivity[2]]
        except KeyError as exc:
            raise ValueError(
                f"Element {element.id} references unknown node id {exc.args[0]}"
            ) from exc

        area = 0.5 * _signed_double_area(n1, n2, n3)
        if abs(area) <= area_tolerance:
            raise ValueError(
                f"Element {element.id} has near-zero area ({area}). Degenerate T3 elements are not allowed."
            )
        if area < 0.0:
            raise ValueError(
                f"Element {element.id} has clockwise node ordering (negative area). Use counter-clockwise connectivity for T3 elements."
            )

        if element.material_id not in material_by_id:
            raise ValueError(
                f"Element {element.id} references unknown material id {element.material_id}"
            )

    for bc in model.boundary_conditions:
        if bc.node_id not in node_by_id:
            raise ValueError(f"Boundary condition references unknown node id {bc.node_id}")
        if bc.dof not in ("ux", "uy"):
            raise ValueError(
                f"Unsupported boundary condition dof '{bc.dof}'. Expected 'ux' or 'uy'."
            )

    for load in model.loads:
        if load.node_id not in node_by_id:
            raise ValueError(f"Load references unknown node id {load.node_id}")
        if load.dof not in ("fx", "fy"):
            raise ValueError(f"Unsupported load dof '{load.dof}'. Expected 'fx' or 'fy'.")


def apply_default_templates_from_physical_groups(
    model: Model,
    gmsh_import: GmshT3ImportResult,
    *,
    default_material: Material | None = None,
    material_by_group_name: dict[str, Material] | None = None,
    clear_existing: bool = True,
) -> TemplateApplicationReport:
    """Apply minimal material/BC/load templates from Gmsh physical group names.

    Naming convention for line physical groups (dimension=1):
    - `BC_FIX` / `BC_FIX_XY`: ux=0, uy=0
    - `BC_FIX_X`: ux=0
    - `BC_FIX_Y`: uy=0
    - `BC_UX=<value>`
    - `BC_UY=<value>`
    - `LOAD_FX=<total_force>`
    - `LOAD_FY=<total_force>`

    Triangle physical groups (dimension=2) are used as material ids.
    """
    base_material = default_material or Material(
        id=1,
        young_modulus=210e9,
        poisson_ratio=0.3,
        plane_stress=True,
    )

    unique_material_ids = sorted({element.material_id for element in model.mesh.elements})
    materials: list[Material] = []
    for material_id in unique_material_ids:
        group_name = gmsh_import.physical_name_by_dim_tag.get((2, material_id), "")
        template = material_by_group_name.get(group_name) if material_by_group_name else None
        material = replace(template or base_material, id=material_id)
        materials.append(material)
    model.materials = materials

    bc_by_node_dof: dict[tuple[int, str], float] = {}
    load_by_node_dof: dict[tuple[int, str], float] = {}

    mapping_entries = build_physical_group_mapping_preview(gmsh_import)
    recognized_group_count = sum(
        1
        for entry in mapping_entries
        if entry.status in ("mapped", "warning") and entry.mapped_to != "unrecognized"
    )

    for physical_tag, node_ids in gmsh_import.line_node_ids_by_physical_tag.items():
        group_name = gmsh_import.physical_name_by_dim_tag.get((1, physical_tag), "")
        rule, value = _parse_group_rule(group_name)
        if not rule:
            continue

        if rule in ("BC_FIX", "BC_FIX_XY"):
            for node_id in node_ids:
                bc_by_node_dof[(node_id, "ux")] = 0.0
                bc_by_node_dof[(node_id, "uy")] = 0.0
        elif rule in ("BC_FIX_X", "BC_UX"):
            prescribed = 0.0 if value is None else value
            for node_id in node_ids:
                bc_by_node_dof[(node_id, "ux")] = prescribed
        elif rule in ("BC_FIX_Y", "BC_UY"):
            prescribed = 0.0 if value is None else value
            for node_id in node_ids:
                bc_by_node_dof[(node_id, "uy")] = prescribed
        elif rule == "LOAD_FX":
            total = 0.0 if value is None else value
            per_node = total / len(node_ids) if node_ids else 0.0
            for node_id in node_ids:
                key = (node_id, "fx")
                load_by_node_dof[key] = load_by_node_dof.get(key, 0.0) + per_node
        elif rule == "LOAD_FY":
            total = 0.0 if value is None else value
            per_node = total / len(node_ids) if node_ids else 0.0
            for node_id in node_ids:
                key = (node_id, "fy")
                load_by_node_dof[key] = load_by_node_dof.get(key, 0.0) + per_node

    boundary_conditions = [
        BoundaryCondition(node_id=node_id, dof=dof, value=value)
        for (node_id, dof), value in sorted(bc_by_node_dof.items())
    ]
    loads = [
        Load(node_id=node_id, dof=dof, value=value)
        for (node_id, dof), value in sorted(load_by_node_dof.items())
    ]

    if clear_existing:
        model.boundary_conditions = boundary_conditions
        model.loads = loads
    else:
        model.boundary_conditions.extend(boundary_conditions)
        model.loads.extend(loads)

    return TemplateApplicationReport(
        material_count=len(model.materials),
        boundary_condition_count=len(model.boundary_conditions),
        load_count=len(model.loads),
        recognized_group_count=recognized_group_count,
        mapping_entries=mapping_entries,
    )
