#pragma once

#include "fem_core/model.hpp"

#include <vector>

namespace fem {

struct NodeDisplacement {
    int node_id;
    double ux;
    double uy;
};

struct ElementStrainStress {
    int element_id;
    std::vector<double> strain;
    std::vector<double> stress;
};

struct StaticSolveSummary {
    std::size_t node_count;
    std::size_t element_count;
    std::size_t total_dof;
    double max_displacement;
};

struct LinearStaticResult {
    std::vector<double> displacements;
    std::vector<double> reactions;
    std::vector<NodeDisplacement> node_displacements;
    std::vector<ElementStrainStress> element_results;
    StaticSolveSummary summary;
};

ModelSummary summarize_model(
    const std::vector<Node>& nodes,
    const std::vector<Element>& elements,
    const std::vector<Material>& materials);

std::vector<double> solve_placeholder(std::size_t dof_count);

LinearStaticResult solve_linear_static(
    const std::vector<Node>& nodes,
    const std::vector<Element>& elements,
    const std::vector<Material>& materials,
    const std::vector<BoundaryCondition>& boundary_conditions,
    const std::vector<Load>& loads,
    double thickness = 1.0,
    double area_tolerance = 1e-14);

}  // namespace fem
