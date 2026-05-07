# FEM Static Solver Baseline (Python + C++)

## Scope
This baseline covers only 2D linear static analysis with T3 elements.
It does not include sparse optimization or advanced material models.
UI and mesh-access capabilities are tracked in `docs/ui_mesh_stage1.md` and `docs/ui_mesh_stage2.md`.

## Backend API
Use the same Python entry for both backends:

```python
from fem_ai_solver.fem.services import solve_linear_static

result_py = solve_linear_static(model, backend="python")
result_cpp = solve_linear_static(model, backend="cpp")
```

Default remains Python:

```python
result_default = solve_linear_static(model)  # same as backend="python"
```

## Result Contract
Both backends are normalized to the same Python result object (`StaticSolveResult`) with fields:
- `displacements`: `np.ndarray`, shape `(2 * node_count,)`
- `reactions`: `np.ndarray`, shape `(2 * node_count,)`
- `node_displacements`: `list[NodeDisplacement]`
- `element_results`: `list[ElementStrainStress]`
- `summary`: `StaticSolveSummary`

`summary` semantics:
- `node_count`: number of model nodes
- `element_count`: number of model elements
- `total_dof`: `2 * node_count`
- `max_displacement`: max nodal displacement magnitude

## Build fem_core (Windows + VS)
From repository root:

```powershell
$pybind = .\.venv\Scripts\python.exe -m pybind11 --cmakedir
cmd /c "`"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul && cmake -S . -B build_nmake -G `"NMake Makefiles`" -Dpybind11_DIR=`"$pybind`""
cmd /c "`"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul && cmake --build build_nmake"
```

The extension is produced under `python/` and can be imported with `PYTHONPATH=python`.

## Test Suite Map
- Python baseline chain:
  - `tests/test_linear_static_solver.py`
  - `tests/test_model_summary.py`
  - `tests/test_t3_element.py`
- C++ unit parity (T3 element layer):
  - `tests/test_cpp_t3_parity.py`
- C++ global parity:
  - `tests/test_cpp_linear_static_parity.py`
- C++ global error paths:
  - `tests/test_cpp_linear_static_errors.py`
- Python backend switch:
  - `tests/test_python_backend_selection.py`
- Smoke tests:
  - `tests/test_smoke_dual_backend.py`
- Backend semantic alignment:
  - `tests/test_backend_semantic_alignment.py`
- Numerical coverage expansion (dual backend):
  - `tests/test_numerical_coverage_dual_backend.py`
- Numerical robustness expansion (dual backend):
  - `tests/test_numerical_robustness_dual_backend.py`
- Mesh import and physical-group template mapping:
  - `tests/test_mesh_importers.py`
- Preprocessing helpers:
  - `tests/test_preprocessing_demo_model.py`
  - `tests/test_preprocessing_model_services.py`
- UI result table helpers:
  - `tests/test_result_table_utils.py`
- Meshing-route placeholders:
  - `tests/test_meshing_placeholders.py`

## Minimal Regression Commands
From repository root:

```powershell
$env:PYTHONPATH='python'
.\.venv\Scripts\python.exe -m pytest tests\test_linear_static_solver.py tests\test_python_backend_selection.py tests\test_cpp_t3_parity.py tests\test_cpp_linear_static_parity.py tests\test_cpp_linear_static_errors.py tests\test_backend_semantic_alignment.py tests\test_smoke_dual_backend.py tests\test_mesh_importers.py tests\test_preprocessing_demo_model.py tests\test_preprocessing_model_services.py tests\test_result_table_utils.py tests\test_meshing_placeholders.py tests\test_numerical_coverage_dual_backend.py tests\test_numerical_robustness_dual_backend.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Automated shortcut:

```powershell
.\scripts\run_min_regression.ps1
.\scripts\run_min_regression.ps1 -All
```

## Current Known Differences
1. `backend="cpp"` requires `fem_core` extension to be built and importable.
   - If extension is unavailable, Python backend still works.
2. Numeric text formatting in some exception messages may differ in decimal formatting details (for example area value precision), while exception type and key message semantics are aligned.
3. In high-condition-number robustness cases, unconstrained DOFs can show near-zero reaction noise differences (typically around `1e-3` absolute) between backends due linear solver implementation differences (`numpy.linalg.solve` vs C++ partial-pivot Gaussian elimination).

## Stable Interface Constraints
Input assumptions intended to remain stable for upper layers:
- `solve_linear_static(model, backend=...)` accepts only T3 elements in the current baseline.
- Global DOF mapping is fixed: for node index `i`, `ux -> 2*i`, `uy -> 2*i+1`.
- Loads are nodal concentrated forces only (`fx`, `fy`).
- Boundary conditions are displacement constraints only (`ux`, `uy`).

Exception semantics intended to remain stable:
- Invalid model/input references -> `ValueError`.
- Underconstrained/singular linear system -> `RuntimeError`.
- Messages keep key semantic phrases used by tests, for example:
  - `clockwise node ordering`
  - `near-zero area`
  - `Load references unknown node id`
  - `Boundary condition references unknown node id`
  - `insufficient displacement constraints`

Result fields intended as long-term GUI/service dependency:
- `displacements`
- `reactions`
- `node_displacements`
- `element_results`
- `summary`

## GUI Integration Boundary (Pre-Integration Contract)
GUI and upper services should rely only on normalized Python-side result objects:
- `StaticSolveResult`
- `NodeDisplacement`
- `ElementStrainStress`
- `StaticSolveSummary`

GUI should not directly depend on backend-private internals:
- C++ `LinearStaticResult` and pybind container layout
- Element-local intermediate matrices (`Ke`, `B`, `D`)
- Backend-specific exception formatting details beyond documented key phrases

Suggested minimum GUI cut-in scope:
- Model solve trigger with backend switch (`python`/`cpp`)
- Nodal displacement table
- Element strain/stress table
- Summary panel (`node_count`, `element_count`, `total_dof`, `max_displacement`)
