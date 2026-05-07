#pragma once

#include "fem_core/model.hpp"

#include <vector>

namespace fem {

std::vector<double> elastic_matrix(const Material& material);

double t3_area(const std::vector<Node>& nodes);

std::vector<double> t3_b_matrix(const std::vector<Node>& nodes);

std::vector<double> t3_stiffness_matrix(
    const std::vector<Node>& nodes,
    const Material& material,
    double thickness = 1.0);

}  // namespace fem
