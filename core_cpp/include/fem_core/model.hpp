#pragma once

#include <string>
#include <vector>

namespace fem {

struct Node {
    int id;
    double x;
    double y;
};

struct Material {
    int id;
    double young_modulus;
    double poisson_ratio;
    bool plane_stress;
};

struct Element {
    int id;
    std::string type;
    std::vector<int> connectivity;
    int material_id;
};

struct BoundaryCondition {
    int node_id;
    std::string dof;
    double value;
};

struct Load {
    int node_id;
    std::string dof;
    double value;
};

struct ModelSummary {
    std::size_t node_count;
    std::size_t element_count;
    std::size_t material_count;
};

}  // namespace fem
