# Third-Party Mesh Sources

This document tracks GitHub-sourced mesh-related code evaluated for RockFEM UI integration.

## Integrated

1. `joelibaceta/triangulator`
- URL: <https://github.com/joelibaceta/triangulator>
- Clone path: `third_party/triangulator`
- Usage: Ear-clipping polygon triangulation adapted into
  `python/fem_ai_solver/mesh/generation/github_ear_clipping.py`
- License signal: `setup.py` metadata declares `MIT`

## Evaluated But Not Integrated

1. `nschloe/meshzoo`
- URL: <https://github.com/nschloe/meshzoo>
- Clone path: `third_party/meshzoo`
- Reason not integrated: README indicates commercial licensing workflow.
  To avoid license ambiguity for RockFEM, code was not wired into runtime logic.

2. `jmespadero/pyDelaunay2D`
- URL: <https://github.com/jmespadero/pyDelaunay2D>
- Clone path: `third_party/pyDelaunay2D`
- Reason not integrated: GPL-3.0 license in `LICENSE` would impose copyleft obligations.

