from __future__ import annotations

import pytest

from fem_ai_solver.fem.model import BoundaryCondition, Element, Load, Material, Node


def _load_fem_core():
    return pytest.importorskip(
        "fem_core",
        reason="fem_core extension is not built in this environment",
    )


def _to_core_nodes(fem_core, nodes: list[Node]):
    values = []
    for node in nodes:
        value = fem_core.Node()
        value.id = node.id
        value.x = node.x
        value.y = node.y
        values.append(value)
    return values


def _to_core_elements(fem_core, elements: list[Element]):
    values = []
    for element in elements:
        value = fem_core.Element()
        value.id = element.id
        value.type = element.type
        value.connectivity = element.connectivity
        value.material_id = element.material_id
        values.append(value)
    return values


def _to_core_materials(fem_core, materials: list[Material]):
    values = []
    for material in materials:
        value = fem_core.Material()
        value.id = material.id
        value.young_modulus = material.young_modulus
        value.poisson_ratio = material.poisson_ratio
        value.plane_stress = material.plane_stress
        values.append(value)
    return values


def _to_core_bcs(fem_core, bcs: list[BoundaryCondition]):
    values = []
    for bc in bcs:
        value = fem_core.BoundaryCondition()
        value.node_id = bc.node_id
        value.dof = bc.dof
        value.value = bc.value
        values.append(value)
    return values


def _to_core_loads(fem_core, loads: list[Load]):
    values = []
    for load in loads:
        value = fem_core.Load()
        value.node_id = load.node_id
        value.dof = load.dof
        value.value = load.value
        values.append(value)
    return values


def _run_cpp(
    fem_core,
    nodes: list[Node],
    elements: list[Element],
    materials: list[Material],
    boundary_conditions: list[BoundaryCondition],
    loads: list[Load],
):
    return fem_core.solve_linear_static(
        _to_core_nodes(fem_core, nodes),
        _to_core_elements(fem_core, elements),
        _to_core_materials(fem_core, materials),
        _to_core_bcs(fem_core, boundary_conditions),
        _to_core_loads(fem_core, loads),
        1.0,
        1e-14,
    )


def _valid_nodes() -> list[Node]:
    return [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=1.0, y=1.0),
        Node(id=4, x=0.0, y=1.0),
    ]


def _valid_elements() -> list[Element]:
    return [
        Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=1),
        Element(id=2, type="T3", connectivity=[1, 3, 4], material_id=1),
    ]


def _valid_materials() -> list[Material]:
    return [Material(id=1, young_modulus=210e9, poisson_ratio=0.3, plane_stress=True)]


def test_cpp_raises_for_insufficient_constraints() -> None:
    fem_core = _load_fem_core()
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=0.0, y=1.0),
    ]
    elements = [Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=1)]
    materials = _valid_materials()
    loads = [Load(node_id=2, dof="fx", value=100.0)]

    with pytest.raises(RuntimeError, match="insufficient displacement constraints"):
        _run_cpp(fem_core, nodes, elements, materials, [], loads)


def test_cpp_raises_for_conflicting_displacement_constraints() -> None:
    fem_core = _load_fem_core()
    nodes = _valid_nodes()
    elements = _valid_elements()
    materials = _valid_materials()
    bcs = [
        BoundaryCondition(node_id=1, dof="ux", value=0.0),
        BoundaryCondition(node_id=1, dof="ux", value=1.0),
    ]

    with pytest.raises(ValueError, match="Conflicting displacement constraints"):
        _run_cpp(fem_core, nodes, elements, materials, bcs, [])


def test_cpp_raises_for_clockwise_element() -> None:
    fem_core = _load_fem_core()
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=0.0, y=1.0),
    ]
    elements = [Element(id=1, type="T3", connectivity=[1, 3, 2], material_id=1)]

    with pytest.raises(ValueError, match="clockwise"):
        _run_cpp(fem_core, nodes, elements, _valid_materials(), [], [])


def test_cpp_raises_for_zero_area_element() -> None:
    fem_core = _load_fem_core()
    nodes = [
        Node(id=1, x=0.0, y=0.0),
        Node(id=2, x=1.0, y=0.0),
        Node(id=3, x=2.0, y=0.0),
    ]
    elements = [Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=1)]

    with pytest.raises(ValueError, match="near-zero area"):
        _run_cpp(fem_core, nodes, elements, _valid_materials(), [], [])


def test_cpp_raises_for_unknown_node_reference_in_element() -> None:
    fem_core = _load_fem_core()
    nodes = _valid_nodes()
    elements = [Element(id=1, type="T3", connectivity=[1, 2, 99], material_id=1)]

    with pytest.raises(ValueError, match="references unknown node id"):
        _run_cpp(fem_core, nodes, elements, _valid_materials(), [], [])


def test_cpp_raises_for_unknown_material_reference() -> None:
    fem_core = _load_fem_core()
    nodes = _valid_nodes()
    elements = [Element(id=1, type="T3", connectivity=[1, 2, 3], material_id=999)]

    with pytest.raises(ValueError, match="references unknown material id"):
        _run_cpp(fem_core, nodes, elements, _valid_materials(), [], [])


def test_cpp_raises_for_invalid_material_parameter() -> None:
    fem_core = _load_fem_core()
    materials = [Material(id=1, young_modulus=0.0, poisson_ratio=0.3, plane_stress=True)]

    with pytest.raises(ValueError, match="young_modulus"):
        _run_cpp(fem_core, _valid_nodes(), _valid_elements(), materials, [], [])


def test_cpp_raises_for_missing_materials() -> None:
    fem_core = _load_fem_core()

    with pytest.raises(ValueError, match="at least one material"):
        _run_cpp(fem_core, _valid_nodes(), _valid_elements(), [], [], [])


def test_cpp_raises_for_unknown_load_node_reference() -> None:
    fem_core = _load_fem_core()
    loads = [Load(node_id=999, dof="fx", value=10.0)]

    with pytest.raises(ValueError, match="Load references unknown node id"):
        _run_cpp(fem_core, _valid_nodes(), _valid_elements(), _valid_materials(), [], loads)


def test_cpp_raises_for_unknown_boundary_node_reference() -> None:
    fem_core = _load_fem_core()
    bcs = [BoundaryCondition(node_id=999, dof="ux", value=0.0)]

    with pytest.raises(ValueError, match="Boundary condition references unknown node id"):
        _run_cpp(fem_core, _valid_nodes(), _valid_elements(), _valid_materials(), bcs, [])


def test_cpp_raises_for_invalid_load_dof() -> None:
    fem_core = _load_fem_core()
    loads = [Load(node_id=1, dof="fz", value=10.0)]

    with pytest.raises(ValueError, match="Unsupported load dof"):
        _run_cpp(fem_core, _valid_nodes(), _valid_elements(), _valid_materials(), [], loads)


def test_cpp_raises_for_invalid_boundary_dof() -> None:
    fem_core = _load_fem_core()
    bcs = [BoundaryCondition(node_id=1, dof="uz", value=0.0)]

    with pytest.raises(ValueError, match="Unsupported boundary condition dof"):
        _run_cpp(fem_core, _valid_nodes(), _valid_elements(), _valid_materials(), bcs, [])
