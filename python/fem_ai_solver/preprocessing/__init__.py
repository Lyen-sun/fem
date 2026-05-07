from fem_ai_solver.fem.model import Model

from .demo_model import (
    apply_left_clamp_right_nodal_force,
    build_stage1_demo_model,
    model_from_t3_mesh,
)
from .model_services import (
    PhysicalGroupMappingEntry,
    TemplateApplicationReport,
    apply_default_templates_from_physical_groups,
    build_physical_group_mapping_preview,
    collect_pre_solve_warnings,
    summarize_mapping_entries,
    summarize_model_state,
    validate_model,
)
from .scene_model import (
    BoundaryDefinition,
    ComponentDef,
    GeometrySetDef,
    LoopDef,
    LoadDefinition,
    RegionDef,
    SceneMeshState,
    SceneProject,
    SceneSelection,
    polygon_area,
    polygon_centroid,
)

__all__ = [
    "Model",
    "PhysicalGroupMappingEntry",
    "TemplateApplicationReport",
    "LoopDef",
    "RegionDef",
    "ComponentDef",
    "GeometrySetDef",
    "LoadDefinition",
    "BoundaryDefinition",
    "SceneMeshState",
    "SceneProject",
    "SceneSelection",
    "apply_default_templates_from_physical_groups",
    "apply_left_clamp_right_nodal_force",
    "build_physical_group_mapping_preview",
    "build_stage1_demo_model",
    "collect_pre_solve_warnings",
    "model_from_t3_mesh",
    "polygon_area",
    "polygon_centroid",
    "summarize_mapping_entries",
    "summarize_model_state",
    "validate_model",
]
