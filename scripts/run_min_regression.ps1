param(
    [switch]$All
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $repoRoot

$python = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw 'Expected virtual environment python at .\.venv\Scripts\python.exe'
}

$env:PYTHONPATH = 'python'

$tests = @(
    'tests/test_linear_static_solver.py',
    'tests/test_python_backend_selection.py',
    'tests/test_cpp_t3_parity.py',
    'tests/test_cpp_linear_static_parity.py',
    'tests/test_cpp_linear_static_errors.py',
    'tests/test_backend_semantic_alignment.py',
    'tests/test_smoke_dual_backend.py',
    'tests/test_mesh_importers.py',
    'tests/test_preprocessing_demo_model.py',
    'tests/test_preprocessing_model_services.py',
    'tests/test_result_table_utils.py',
    'tests/test_meshing_placeholders.py',
    'tests/test_numerical_coverage_dual_backend.py',
    'tests/test_numerical_robustness_dual_backend.py',
    'tests/test_geotech_template_workflow.py'
)

& $python -m pytest @tests -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($All) {
    & $python -m pytest -q
    exit $LASTEXITCODE
}


