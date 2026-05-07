# FEM AI Solver

一个面�?2D 有限元分析的工程软件原型项目，目标是逐步演进为具备以下能力的桌面应用�?
- 基础 FEM 建模、装配、求解与后处�?- PySide6 图形界面
- Python + C++ 混合高性能计算内核
- AI 辅助题目解析、建模建议与结果解释

## 当前工程目标

当前版本优先解决三个问题�?
1. 把工程骨架搭对，避免后期重构成本过高
2. 统一 FEM 数据模型，支撑多单元类型扩展
3. �?C++ 求解核心�?Python GUI/AI 之间建立稳定接口

## 建议开发顺�?
1. `T3 + 平面应力/平面应变 + 线弹�?+ 静力学`
2. `Q4`
3. `T6`
4. 稀疏组装与求解器升�?5. GUI 交互建模与后处理
6. AI 辅助模块接入

## 目录结构

```text
fem_ai_solver/
├── core_cpp/                # C++ FEM 核心�?pybind11 绑定
├── docs/                    # 架构文档与开发路�?├── python/
�?  ├── fem_ai_solver/
�?  �?  ├── fem/             # Python 侧领域模型与流程编排
�?  �?  ├── ui/              # GUI
�?  �?  ├── ai/              # AI 模块
�?  �?  └── app.py           # 桌面应用入口
├── tests/                   # 单元测试/集成测试
├── CMakeLists.txt
└── pyproject.toml
```

## 近期里程�?
- M1: 单元级验�?- M2: 整体模型求解跑�?- M3: GUI 可视�?- M4: AI 辅助建模与解�?
详见 [docs/architecture.md](I:\wm\fem_2\docs\architecture.md) �?[docs/roadmap.md](I:\wm\fem_2\docs\roadmap.md)�?
## Backend Baseline Notes
- Baseline and regression guidance: `docs/backend_baseline.md`
- Minimal dual-backend example: `examples/minimal_linear_static_dual_backend.py`

## Extended Dual-Backend Example
- `examples/dual_backend_multimaterial_compare.py` demonstrates a larger multi-element, multi-material case and compares Python/C++ backend outputs.

## UI + Mesh Stage1
- Stage1 UI and mesh import contract: `docs/ui_mesh_stage1.md`
- Launch app: `$env:PYTHONPATH='python'; .\.venv\Scripts\python.exe -m fem_ai_solver.app`
- Stage2 UI + preprocessing + mesh access notes: `docs/ui_mesh_stage2.md`

