from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fem_ai_solver.fem.model import BoundaryCondition, Element, Load, Material, Model, Node
from fem_ai_solver.fem.results import (
    ElementStrainStress,
    NodeDisplacement,
    StaticSolveResult,
    StaticSolveSummary,
)

try:
    import fem_core as _fem_core
except ImportError as exc:  # pragma: no cover - exercised in fallback environments
    _fem_core = None
    _FEM_CORE_IMPORT_ERROR = exc
else:
    _FEM_CORE_IMPORT_ERROR = None


FloatArray = NDArray[np.float64]
_AREA_TOLERANCE = 1e-14


@dataclass(slots=True)
class _ResolvedElement:
    element: Element
    nodes: list[Node]
    dof_indices: list[int]
    material: Material


def summarize_model(model: Model) -> dict[str, int]:
    return {
        "node_count": len(model.mesh.nodes),
        "element_count": len(model.mesh.elements),
        "material_count": len(model.materials),
        "boundary_condition_count": len(model.boundary_conditions),
        "load_count": len(model.loads),
    }


def model_to_payload(model: Model) -> dict:
    return asdict(model)


def _require_valid_material(material: Material) -> None:
    if material.young_modulus <= 0.0:
        raise ValueError("young_modulus must be positive")
    if material.poisson_ratio <= -1.0 or material.poisson_ratio >= 0.5:
        raise ValueError("poisson_ratio must be in (-1.0, 0.5)")


def _require_valid_t3(nodes: list[Node]) -> None:
    if len(nodes) != 3:
        raise ValueError("T3 element requires exactly 3 nodes")


def _to_core_node(node: Node):
    result = _fem_core.Node()
    result.id = node.id
    result.x = node.x
    result.y = node.y
    return result


def _to_core_material(material: Material):
    result = _fem_core.Material()
    result.id = material.id
    result.young_modulus = material.young_modulus
    result.poisson_ratio = material.poisson_ratio
    result.plane_stress = material.plane_stress
    return result


def _to_core_element(element: Element):
    result = _fem_core.Element()
    result.id = element.id
    result.type = element.type
    result.connectivity = element.connectivity
    result.material_id = element.material_id
    return result


def _to_core_boundary_condition(boundary_condition: BoundaryCondition):
    result = _fem_core.BoundaryCondition()
    result.node_id = boundary_condition.node_id
    result.dof = boundary_condition.dof
    result.value = boundary_condition.value
    return result


def _to_core_load(load: Load):
    result = _fem_core.Load()
    result.node_id = load.node_id
    result.dof = load.dof
    result.value = load.value
    return result


def _signed_double_area(nodes: list[Node]) -> float:
    n1, n2, n3 = nodes
    return (n2.x - n1.x) * (n3.y - n1.y) - (n3.x - n1.x) * (n2.y - n1.y)


def _t3_coefficients(nodes: list[Node]) -> tuple[float, list[float], list[float]]:
    _require_valid_t3(nodes)
    two_area = _signed_double_area(nodes)
    if two_area <= 0.0:
        raise ValueError("T3 area must be positive for counter-clockwise node ordering")

    n1, n2, n3 = nodes
    beta = [n2.y - n3.y, n3.y - n1.y, n1.y - n2.y]
    gamma = [n3.x - n2.x, n1.x - n3.x, n2.x - n1.x]
    return two_area, beta, gamma


def _elastic_matrix_py(material: Material) -> list[list[float]]:
    _require_valid_material(material)
    e = material.young_modulus
    nu = material.poisson_ratio

    if material.plane_stress:
        factor = e / (1.0 - nu * nu)
        return [
            [factor, factor * nu, 0.0],
            [factor * nu, factor, 0.0],
            [0.0, 0.0, factor * (1.0 - nu) * 0.5],
        ]

    factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return [
        [factor * (1.0 - nu), factor * nu, 0.0],
        [factor * nu, factor * (1.0 - nu), 0.0],
        [0.0, 0.0, factor * (1.0 - 2.0 * nu) * 0.5],
    ]


def _t3_b_matrix_py(nodes: list[Node]) -> list[list[float]]:
    two_area, beta, gamma = _t3_coefficients(nodes)
    inv_two_area = 1.0 / two_area
    return [
        [beta[0] * inv_two_area, 0.0, beta[1] * inv_two_area, 0.0, beta[2] * inv_two_area, 0.0],
        [0.0, gamma[0] * inv_two_area, 0.0, gamma[1] * inv_two_area, 0.0, gamma[2] * inv_two_area],
        [gamma[0] * inv_two_area, beta[0] * inv_two_area, gamma[1] * inv_two_area, beta[1] * inv_two_area, gamma[2] * inv_two_area, beta[2] * inv_two_area],
    ]


def elastic_matrix(material: Material) -> list[list[float]]:
    if _fem_core is not None:
        flat = _fem_core.elastic_matrix(_to_core_material(material))
        return [flat[0:3], flat[3:6], flat[6:9]]
    return _elastic_matrix_py(material)


def t3_area(nodes: list[Node]) -> float:
    if _fem_core is not None:
        return _fem_core.t3_area([_to_core_node(node) for node in nodes])
    two_area, _, _ = _t3_coefficients(nodes)
    return 0.5 * two_area


def t3_b_matrix(nodes: list[Node]) -> list[list[float]]:
    if _fem_core is not None:
        flat = _fem_core.t3_b_matrix([_to_core_node(node) for node in nodes])
        return [flat[0:6], flat[6:12], flat[12:18]]
    return _t3_b_matrix_py(nodes)


def _t3_stiffness_matrix_py(
    nodes: list[Node],
    material: Material,
    thickness: float = 1.0,
) -> list[list[float]]:
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")

    d = _elastic_matrix_py(material)
    b = _t3_b_matrix_py(nodes)
    area = t3_area(nodes)

    db = [[0.0 for _ in range(6)] for _ in range(3)]
    for i in range(3):
        for j in range(6):
            db[i][j] = sum(d[i][k] * b[k][j] for k in range(3))

    scale = thickness * area
    ke = [[0.0 for _ in range(6)] for _ in range(6)]
    for i in range(6):
        for j in range(6):
            ke[i][j] = scale * sum(b[k][i] * db[k][j] for k in range(3))
    return ke


def t3_stiffness_matrix(
    nodes: list[Node],
    material: Material,
    thickness: float = 1.0,
) -> list[list[float]]:
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")

    if _fem_core is not None:
        flat = _fem_core.t3_stiffness_matrix(
            [_to_core_node(node) for node in nodes],
            _to_core_material(material),
            thickness,
        )
        return [flat[index:index + 6] for index in range(0, 36, 6)]

    return _t3_stiffness_matrix_py(nodes, material, thickness)


def _build_node_index(model: Model) -> tuple[dict[int, Node], dict[int, int]]:
    if not model.mesh.nodes:
        raise ValueError("model must contain at least one node")

    node_by_id: dict[int, Node] = {}
    node_index_by_id: dict[int, int] = {}
    for index, node in enumerate(model.mesh.nodes):
        if node.id in node_by_id:
            raise ValueError(f"duplicate node id detected: {node.id}")
        node_by_id[node.id] = node
        node_index_by_id[node.id] = index

    return node_by_id, node_index_by_id


def _build_material_index(model: Model) -> dict[int, Material]:
    if not model.materials:
        raise ValueError("model must contain at least one material")

    material_by_id: dict[int, Material] = {}
    for material in model.materials:
        _require_valid_material(material)
        if material.id in material_by_id:
            raise ValueError(f"duplicate material id detected: {material.id}")
        material_by_id[material.id] = material

    return material_by_id


def _element_dof_indices(node_indices: list[int]) -> list[int]:
    """Map node indices to element DOFs using [n1x, n1y, n2x, n2y, n3x, n3y].

    For node index i in model.mesh.nodes, the global DOF mapping is:
    dof_x = 2 * i, dof_y = 2 * i + 1.
    """
    dof_indices: list[int] = []
    for node_index in node_indices:
        dof_indices.extend([2 * node_index, 2 * node_index + 1])
    return dof_indices


def _validate_t3_geometry(nodes: list[Node], element_id: int, area_tolerance: float) -> None:
    _require_valid_t3(nodes)
    two_area = _signed_double_area(nodes)
    area = 0.5 * two_area

    if abs(area) <= area_tolerance:
        raise ValueError(
            f"Element {element_id} has near-zero area ({area}). Degenerate T3 elements are not allowed."
        )
    if area < 0.0:
        raise ValueError(
            f"Element {element_id} has clockwise node ordering (negative area). "
            "Use counter-clockwise connectivity for T3 elements."
        )


def _resolve_elements(
    model: Model,
    node_by_id: dict[int, Node],
    node_index_by_id: dict[int, int],
    material_by_id: dict[int, Material],
    area_tolerance: float,
) -> list[_ResolvedElement]:
    if not model.mesh.elements:
        raise ValueError("model must contain at least one element")

    resolved_elements: list[_ResolvedElement] = []
    for element in model.mesh.elements:
        if element.type != "T3":
            raise ValueError(
                f"Element {element.id} has unsupported type '{element.type}'. "
                "solve_linear_static currently supports only T3 elements."
            )
        if len(element.connectivity) != 3:
            raise ValueError(
                f"Element {element.id} must have exactly 3 node ids in connectivity for T3."
            )

        nodes: list[Node] = []
        node_indices: list[int] = []
        for node_id in element.connectivity:
            node = node_by_id.get(node_id)
            if node is None:
                raise ValueError(f"Element {element.id} references unknown node id {node_id}")
            nodes.append(node)
            node_indices.append(node_index_by_id[node_id])

        _validate_t3_geometry(nodes, element.id, area_tolerance)

        material = material_by_id.get(element.material_id)
        if material is None:
            raise ValueError(
                f"Element {element.id} references unknown material id {element.material_id}"
            )

        resolved_elements.append(
            _ResolvedElement(
                element=element,
                nodes=nodes,
                dof_indices=_element_dof_indices(node_indices),
                material=material,
            )
        )

    return resolved_elements


def _assemble_global_stiffness(
    ndof: int,
    resolved_elements: list[_ResolvedElement],
    thickness: float,
) -> FloatArray:
    """Assemble dense global stiffness matrix K from element stiffness matrices."""
    if ndof <= 0:
        raise ValueError("model has no effective degrees of freedom")

    k_global = np.zeros((ndof, ndof), dtype=np.float64)
    for resolved in resolved_elements:
        ke = np.asarray(
            t3_stiffness_matrix(resolved.nodes, resolved.material, thickness=thickness),
            dtype=np.float64,
        )
        if ke.shape != (6, 6):
            raise RuntimeError(
                f"Element {resolved.element.id} stiffness has invalid shape {ke.shape}, expected (6, 6)."
            )

        dofs = resolved.dof_indices
        k_global[np.ix_(dofs, dofs)] += ke

    return k_global


def _build_global_force_vector(
    loads: list[Load],
    node_index_by_id: dict[int, int],
    ndof: int,
) -> FloatArray:
    """Assemble global nodal force vector F from concentrated loads."""
    force = np.zeros(ndof, dtype=np.float64)

    for load in loads:
        node_index = node_index_by_id.get(load.node_id)
        if node_index is None:
            raise ValueError(f"Load references unknown node id {load.node_id}")

        if load.dof == "fx":
            dof = 2 * node_index
        elif load.dof == "fy":
            dof = 2 * node_index + 1
        else:
            raise ValueError(f"Unsupported load dof '{load.dof}'. Expected 'fx' or 'fy'.")

        force[dof] += load.value

    return force


def _collect_prescribed_displacements(
    boundary_conditions: list[BoundaryCondition],
    node_index_by_id: dict[int, int],
) -> dict[int, float]:
    """Collect prescribed displacement DOFs for Dirichlet boundary conditions."""
    prescribed: dict[int, float] = {}

    for bc in boundary_conditions:
        node_index = node_index_by_id.get(bc.node_id)
        if node_index is None:
            raise ValueError(f"Boundary condition references unknown node id {bc.node_id}")

        if bc.dof == "ux":
            dof = 2 * node_index
        elif bc.dof == "uy":
            dof = 2 * node_index + 1
        else:
            raise ValueError(f"Unsupported boundary condition dof '{bc.dof}'. Expected 'ux' or 'uy'.")

        value = float(bc.value)
        existing = prescribed.get(dof)
        if existing is not None and not np.isclose(existing, value):
            raise ValueError(
                f"Conflicting displacement constraints on global dof {dof}: {existing} vs {value}."
            )
        prescribed[dof] = value

    return prescribed


def _apply_displacement_bcs(
    k_global: FloatArray,
    force: FloatArray,
    prescribed: dict[int, float],
) -> tuple[FloatArray, FloatArray]:
    """Apply u[dof]=value constraints using row/column elimination in-place on copies."""
    k_modified = k_global.copy()
    f_modified = force.copy()

    for dof, value in sorted(prescribed.items()):
        if dof < 0 or dof >= k_modified.shape[0]:
            raise ValueError(f"Boundary condition DOF {dof} is out of range")

        column = k_modified[:, dof].copy()
        f_modified -= column * value

        k_modified[dof, :] = 0.0
        k_modified[:, dof] = 0.0
        k_modified[dof, dof] = 1.0
        f_modified[dof] = value

    return k_modified, f_modified


def _solve_linear_system(k_modified: FloatArray, f_modified: FloatArray) -> FloatArray:
    """Solve KU=F and report FEM-friendly diagnostics for singular systems."""
    if k_modified.size == 0:
        raise ValueError("model has no effective degrees of freedom")

    try:
        displacement = np.linalg.solve(k_modified, f_modified)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Failed to solve the linear system. Possible causes: insufficient displacement "
            "constraints (rigid body modes), invalid/degenerate element geometry, or "
            "inconsistent material parameters."
        ) from exc

    if not np.all(np.isfinite(displacement)):
        raise RuntimeError("Linear solver produced non-finite displacement values.")

    return displacement.astype(np.float64, copy=False)


def _recover_element_results(
    resolved_elements: list[_ResolvedElement],
    displacement: FloatArray,
) -> list[ElementStrainStress]:
    """Recover constant strain/stress for each T3 element from solved displacement U."""
    element_results: list[ElementStrainStress] = []

    for resolved in resolved_elements:
        b_matrix = np.asarray(t3_b_matrix(resolved.nodes), dtype=np.float64)
        d_matrix = np.asarray(elastic_matrix(resolved.material), dtype=np.float64)

        if b_matrix.shape != (3, 6):
            raise RuntimeError(
                f"Element {resolved.element.id} B matrix has invalid shape {b_matrix.shape}, expected (3, 6)."
            )
        if d_matrix.shape != (3, 3):
            raise RuntimeError(
                f"Element {resolved.element.id} D matrix has invalid shape {d_matrix.shape}, expected (3, 3)."
            )

        ue = displacement[np.asarray(resolved.dof_indices, dtype=np.int64)]
        strain = b_matrix @ ue
        stress = d_matrix @ strain

        if not (np.all(np.isfinite(strain)) and np.all(np.isfinite(stress))):
            raise RuntimeError(
                f"Element {resolved.element.id} produced non-finite strain/stress during recovery."
            )

        element_results.append(
            ElementStrainStress(
                element_id=resolved.element.id,
                strain=tuple(float(value) for value in strain.tolist()),
                stress=tuple(float(value) for value in stress.tolist()),
            )
        )

    return element_results


def _build_node_displacements(
    nodes: list[Node],
    node_index_by_id: dict[int, int],
    displacement: FloatArray,
) -> list[NodeDisplacement]:
    node_results: list[NodeDisplacement] = []
    for node in nodes:
        node_index = node_index_by_id[node.id]
        node_results.append(
            NodeDisplacement(
                node_id=node.id,
                ux=float(displacement[2 * node_index]),
                uy=float(displacement[2 * node_index + 1]),
            )
        )
    return node_results


def _solve_linear_static_cpp(
    model: Model,
    thickness: float,
    area_tolerance: float,
) -> StaticSolveResult:
    if _fem_core is None:
        raise RuntimeError(
            "backend='cpp' requires the fem_core extension to be built and importable"
        ) from _FEM_CORE_IMPORT_ERROR

    cpp_result = _fem_core.solve_linear_static(
        [_to_core_node(node) for node in model.mesh.nodes],
        [_to_core_element(element) for element in model.mesh.elements],
        [_to_core_material(material) for material in model.materials],
        [_to_core_boundary_condition(bc) for bc in model.boundary_conditions],
        [_to_core_load(load) for load in model.loads],
        thickness,
        area_tolerance,
    )

    displacement = np.asarray(cpp_result.displacements, dtype=np.float64)
    reactions = np.asarray(cpp_result.reactions, dtype=np.float64)

    element_results = [
        ElementStrainStress(
            element_id=int(item.element_id),
            strain=tuple(float(value) for value in item.strain),
            stress=tuple(float(value) for value in item.stress),
        )
        for item in cpp_result.element_results
    ]

    if hasattr(cpp_result, "node_displacements"):
        node_results = [
            NodeDisplacement(
                node_id=int(item.node_id),
                ux=float(item.ux),
                uy=float(item.uy),
            )
            for item in cpp_result.node_displacements
        ]
    else:
        node_index_by_id = {node.id: index for index, node in enumerate(model.mesh.nodes)}
        node_results = _build_node_displacements(model.mesh.nodes, node_index_by_id, displacement)

    if hasattr(cpp_result, "summary"):
        summary = StaticSolveSummary(
            node_count=int(cpp_result.summary.node_count),
            element_count=int(cpp_result.summary.element_count),
            total_dof=int(cpp_result.summary.total_dof),
            max_displacement=float(cpp_result.summary.max_displacement),
        )
    else:
        nodal_magnitude = np.sqrt(displacement[0::2] ** 2 + displacement[1::2] ** 2)
        summary = StaticSolveSummary(
            node_count=len(model.mesh.nodes),
            element_count=len(model.mesh.elements),
            total_dof=2 * len(model.mesh.nodes),
            max_displacement=float(np.max(nodal_magnitude)) if nodal_magnitude.size else 0.0,
        )

    return StaticSolveResult(
        displacements=displacement,
        reactions=reactions,
        node_displacements=node_results,
        element_results=element_results,
        summary=summary,
    )


def solve_linear_static(
    model: Model,
    thickness: float = 1.0,
    area_tolerance: float = _AREA_TOLERANCE,
    backend: Literal["python", "cpp"] = "python",
) -> StaticSolveResult:
    """Solve 2D linear static elasticity with T3 elements.

    DOF numbering rule:
    - For node index i in model.mesh.nodes: dof_x = 2*i, dof_y = 2*i + 1.
    - Element local DOFs follow [n1x, n1y, n2x, n2y, n3x, n3y].
    """
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if area_tolerance <= 0.0:
        raise ValueError("area_tolerance must be positive")
    if backend not in ("python", "cpp"):
        raise ValueError("backend must be either 'python' or 'cpp'")

    if backend == "cpp":
        return _solve_linear_static_cpp(model, thickness, area_tolerance)

    node_by_id, node_index_by_id = _build_node_index(model)
    if len(node_by_id) < 3:
        raise ValueError("model must contain at least 3 nodes for T3 analysis")

    material_by_id = _build_material_index(model)
    resolved_elements = _resolve_elements(
        model,
        node_by_id,
        node_index_by_id,
        material_by_id,
        area_tolerance,
    )

    ndof = 2 * len(model.mesh.nodes)
    k_original = _assemble_global_stiffness(ndof, resolved_elements, thickness)
    f_original = _build_global_force_vector(model.loads, node_index_by_id, ndof)

    prescribed = _collect_prescribed_displacements(model.boundary_conditions, node_index_by_id)
    k_modified, f_modified = _apply_displacement_bcs(k_original, f_original, prescribed)

    displacement = _solve_linear_system(k_modified, f_modified)
    reactions = k_original @ displacement - f_original

    element_results = _recover_element_results(resolved_elements, displacement)
    node_results = _build_node_displacements(model.mesh.nodes, node_index_by_id, displacement)

    nodal_magnitude = np.sqrt(displacement[0::2] ** 2 + displacement[1::2] ** 2)
    summary = StaticSolveSummary(
        node_count=len(model.mesh.nodes),
        element_count=len(model.mesh.elements),
        total_dof=ndof,
        max_displacement=float(np.max(nodal_magnitude)) if nodal_magnitude.size else 0.0,
    )

    return StaticSolveResult(
        displacements=displacement,
        reactions=reactions,
        node_displacements=node_results,
        element_results=element_results,
        summary=summary,
    )

