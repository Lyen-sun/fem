# UI + Mesh Stage1 Plan and Contract

## Layered Architecture (Stage1)
- Model layer: `python/fem_ai_solver/fem/model.py`
  - Owns `Node/Element/Material/BoundaryCondition/Load/Mesh/Model`.
- Solver service layer: `python/fem_ai_solver/fem/services.py`
  - Owns `solve_linear_static(model, backend=...)` and backend dispatch.
- Result layer: `python/fem_ai_solver/fem/results.py`
  - Owns `StaticSolveResult` and related typed result objects.
- UI and visualization layer: `python/fem_ai_solver/ui/`
  - Consumes only model/service/result Python APIs.
  - Must not call `fem_core` pybind objects directly.
- Mesh generation/import access layer: `python/fem_ai_solver/mesh/`, `python/fem_ai_solver/preprocessing/`
  - Stage1 focuses on mesh import + model assembly helpers, not full mesher implementation.

## Module Boundary (Minimal Additions)
- `python/fem_ai_solver/preprocessing/demo_model.py`
  - Build demo model.
  - Build `Model` from imported T3 mesh.
  - Attach deterministic default boundary/load case for UI solve demo.
- `python/fem_ai_solver/mesh/importers.py`
  - `load_t3_mesh_from_json(...)`
  - `load_t3_mesh_from_gmsh_msh(...)` (minimal ASCII .msh importer for type-2 triangles)
- `python/fem_ai_solver/ui/mesh_canvas.py`
  - Draw undeformed/deformed triangular mesh.
  - Node/element id overlays.
- `python/fem_ai_solver/ui/main_window.py`
  - Demo model load, mesh import, solve trigger, summary panel, view controls.

## Stage1 UI Capability
- Load or build demo model.
- Import T3 mesh (`.json`, `.msh`) and auto-wrap into a solvable model template.
- Render initial mesh.
- Toggle node ids / element ids.
- Solve with backend switch (`python` / `cpp`).
- Show summary fields from `StaticSolveResult.summary`.
- Overlay deformed mesh with user-defined scale factor.

## External Mesher Integration Evaluation

### Option A: Gmsh (recommended first integration)
Pros:
- Strong ecosystem and mature CLI/tooling.
- Easy offline generation from `.geo`/CAD and export to `.msh`.
- Supports physical groups and future expansion (boundaries, materials, multi-region).
- Fits current T3 solver by importing 3-node triangles directly.

Cons:
- Native Python API dependency is heavier than pure-file import.
- Rich feature set increases integration surface if over-adopted too early.

### Option B: Triangle / MeshPy
Pros:
- Lightweight focus on 2D triangular meshing.
- Good fit for constrained Delaunay workflows.
- Cleaner dependency for pure-2D meshing pipelines.

Cons:
- Weaker end-to-end CAD/geometry workflow than Gmsh.
- Less convenient for mixed pre/post toolchain use compared with `.msh` ecosystem.

### Stage1 Recommendation
1. Keep current solver untouched.
2. Start from file-based Gmsh `.msh` import path (already added).
3. Add physical-group mapping in next step for BC/material region tagging.
4. Re-evaluate direct Gmsh API only after file-based flow is stable.

## Short-Term Engineering Route (next 1-2 iterations)
1. Stabilize UI data flow contract on `Model -> solve_linear_static -> StaticSolveResult`.
2. Add import-side validation/reporting (unsupported element types, empty sections, tag warnings).
3. Add region/boundary tagging map from imported mesh metadata to materials/BC templates.
4. Keep GUI tables simple first (node displacement table, element stress/strain table).

## Mid-Term Technical Highlight Route (meshing as differentiator)

### Route 1: Constraint-aware 2D Delaunay pipeline
- Focus: constrained Delaunay triangulation with boundary preservation and region markers.
- Why now: current solver is T3-first and already validates geometry/ordering, so this is a direct fit.
- Increment plan:
  1. boundary segment + hole + region schema
  2. triangulation adapter
  3. quality report and solver-ready model export

### Route 2: Size-field-driven adaptive T3 meshing
- Focus: local size control around stress concentration regions.
- Why now: solver already has strain/stress element results; they can seed refinement indicators.
- Increment plan:
  1. size field schema and visualization
  2. import/refine/re-solve loop
  3. quality + error indicator dashboard

Both routes can evolve without breaking the current backend contract.
