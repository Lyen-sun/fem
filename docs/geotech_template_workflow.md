# 2D Geotechnical Template Workflow

This module adds an end-to-end **plane-strain geotechnical template workflow** on top of the existing dual-backend FEM solver.

## 1) Goal

Input layered cross-section data and automatically run:

1. Layer geometry generation
2. Automatic T3 mesh generation
3. Template boundary/load assignment
4. Linear static solve (`backend="python"` or `"cpp"`)
5. Output extraction (`Ux`, `Uy`, `Von Mises`, A/B points)

## 2) Python API

```python
from fem_ai_solver.geotech import (
    build_template_input_from_text_description,
    run_geotech_template_workflow,
)

template = build_template_input_from_text_description(
    """
    layer Fill: top=[(0,0),(30,0)] bottom=[(0,-4),(30,-6)] E=2.0e7 nu=0.30
    layer Clay: top=[(0,-4),(30,-6)] bottom=[(0,-12),(30,-12)] E=3.7e7 nu=0.29
    Q=35
    mesh_size=1.0
    point A=(2,-3.5)
    point B=(20,-5.0)
    """,
    backend="python",
)

result = run_geotech_template_workflow(template, output_dir="output/geotech_case")
```

## 3) CSV Input Format

`load_geotech_layers_from_csv(...)` expects rows with:

- Required columns: `layer`, `boundary`, `x`, `y`
- Optional columns: `point_order`, `material_id`, `young_modulus`, `poisson_ratio`, `plane_stress`

`boundary` must be `top` or `bottom`.

Example:

```csv
layer,boundary,point_order,x,y,material_id,young_modulus,poisson_ratio,plane_stress
Fill,top,1,0,0,1,2.0e7,0.30,false
Fill,top,2,30,0,1,2.0e7,0.30,false
Fill,bottom,1,0,-4,1,2.0e7,0.30,false
Fill,bottom,2,30,-6,1,2.0e7,0.30,false
```

## 4) Default Geotechnical Template BC/Load

The workflow applies:

- Bottom boundary: `ux=0, uy=0` (configurable)
- Left/right boundaries: `ux=0` (configurable)
- Top line load `Q` (converted to nodal `fy` equivalent loads)

## 5) Output Files

When `output_dir` is provided:

- `node_displacements.csv`
- `element_results.csv`
- `point_displacements.csv`
- `ux_cloud.svg`
- `uy_cloud.svg`
- `von_mises_cloud.svg`
- `summary.json`

## 6) Optional AI-Assisted Input

`fem_ai_solver.ai.geotech_assistant` provides:

- `parse_geotech_text_description(text)` for rule-based text parsing
- `detect_geotech_from_image(path)` placeholder extension point for future image recognition integration

## 7) Example Script

Run:

```powershell
$env:PYTHONPATH="python"
.\.venv\Scripts\python.exe examples\geotech_template_workflow_demo.py --backend python
```

Use C++ backend if `fem_core` is built:

```powershell
$env:PYTHONPATH="python"
.\.venv\Scripts\python.exe examples\geotech_template_workflow_demo.py --backend cpp
```

