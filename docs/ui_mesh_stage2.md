# UI + Preprocessing + Mesh Access Stage2+

## 1. Layer Boundary (kept stable)
- Model layer:
  - `python/fem_ai_solver/fem/model.py`
- Solver service layer:
  - `python/fem_ai_solver/fem/services.py`
  - `solve_linear_static(model, backend=...)`
- Result layer:
  - `python/fem_ai_solver/fem/results.py`
  - `StaticSolveResult` and summary/element/node result records
- UI layer:
  - `python/fem_ai_solver/ui/`
  - Depends only on Python model/service/result objects
  - Does **not** depend on raw `fem_core` pybind objects
- Mesh import / preprocessing layer:
  - `python/fem_ai_solver/mesh/`
  - `python/fem_ai_solver/preprocessing/`

## 2. Stage2+ UI capability additions
- Node displacement table (`node_id`, `ux`, `uy`)
  - basic filter
  - sortable columns
  - CSV export
- Element strain/stress table (`ex`, `ey`, `gxy`, `sx`, `sy`, `txy`)
  - basic filter
  - sortable columns
  - CSV export
- Table-to-mesh linkage (minimal)
  - selecting node rows highlights nodes in mesh view
  - selecting element rows highlights elements in mesh view
- Model status panel:
  - node count
  - element count
  - material count
  - current backend
- Mesh overlays:
  - constrained-node markers
  - nodal load arrows
- Existing stage1/stage2 capabilities retained:
  - demo model
  - mesh import
  - solve trigger
  - summary
  - deformed mesh scale control

## 3. Physical groups mapping transparency and control
Implemented now:
- Import-time mapping preview table:
  - dimension/tag/name
  - mapped target
  - status (`mapped` / `warning` / `unrecognized`)
  - affected item count
  - explanatory note
- Mapping summary counters:
  - total / mapped / warning / unrecognized group count
- Import confirmation step:
  - user can choose whether to apply recognized mapping templates
- Non-mapped or warning groups are logged clearly in UI log panel

## 4. Gmsh physical group minimal mapping (stage2+)
Implemented input:
- ASCII `.msh` v2
- Optional `$PhysicalNames`
- Line elements (type 1) for template groups
- Triangle elements (type 2) for T3 mesh

Implemented mapping rules:
- Triangle physical group tag -> `element.material_id`
- Dimension-2 physical names are available for material template selection
- Dimension-1 physical names map to template rules:
  - `BC_FIX` / `BC_FIX_XY`
  - `BC_FIX_X`
  - `BC_FIX_Y`
  - `BC_UX=<value>`
  - `BC_UY=<value>`
  - `LOAD_FX=<total>`
  - `LOAD_FY=<total>`
- Load templates distribute total force evenly to nodes in the matched line group

## 5. Model service additions
- `validate_model(model, area_tolerance=...)`
  - checks node/element/material references and T3 geometry orientation
- `summarize_model_state(model, backend=...)`
  - counts + constrained/loaded node stats
- `build_physical_group_mapping_preview(gmsh_import)`
  - structured mapping preview entries
- `summarize_mapping_entries(entries)`
  - aggregate mapping-status counters
- `collect_pre_solve_warnings(model)`
  - lightweight pre-solve warning hints
- `apply_default_templates_from_physical_groups(model, gmsh_import, ...)`
  - applies minimal material/boundary/load templates from physical groups
  - returns report including mapping entries

## 6. Current limitations
- Gmsh binary format: not supported
- Gmsh v4 entity-block parsing: not supported yet
- High-order elements: not supported
- Physical-group mapping is convention-driven by group naming and intentionally minimal
- Mapping confirmation is yes/no only (no fine-grained per-group editor yet)

## 7. Mid-term meshing highlight routes

### Route A: constrained Delaunay triangulation
Connection to current solver:
- outputs T3 mesh directly
- can reuse the same `Model -> solve_linear_static` pipeline

Needed core structures:
- `BoundaryLoop` / `SegmentConstraint`
- `RegionMarker` (for material partition)
- triangulation quality report

Suitable module entry:
- `python/fem_ai_solver/mesh/generation/constrained_delaunay.py`

Why suitable now:
- current solver and UI already centered on T3
- physical-group/material-partition routing is now in place

### Route B: size-field-driven adaptive T3 meshing
Connection to current solver:
- element strain/stress already available for refinement indicators
- iterative loop can remain in Python service orchestration

Needed core structures:
- `SizeField` (`h(x, y)`)
- `ElementIndicator` (error/quality measure)
- `RemeshHistory` (old-new element relation)

Suitable module entry:
- `python/fem_ai_solver/mesh/adaptive/refinement.py`

Why suitable now:
- stage2+ has result tables, mesh highlights, and import-preprocess chain
- can show clear value by closed-loop solve-refine-resolve workflow
