#include "fem_core/solver.hpp"

#include "fem_core/elements.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

constexpr double kSingularTolerance = 1e-14;

struct ResolvedElement {
    int element_id;
    std::vector<fem::Node> nodes;
    std::array<int, 6> dof_indices;
    const fem::Material* material;
};

int displacement_dof(const std::string& dof, int node_index) {
    if (dof == "ux") {
        return 2 * node_index;
    }
    if (dof == "uy") {
        return 2 * node_index + 1;
    }
    throw std::invalid_argument(
        "Unsupported boundary condition dof '" + dof + "'. Expected 'ux' or 'uy'.");
}

int force_dof(const std::string& dof, int node_index) {
    if (dof == "fx") {
        return 2 * node_index;
    }
    if (dof == "fy") {
        return 2 * node_index + 1;
    }
    throw std::invalid_argument(
        "Unsupported load dof '" + dof + "'. Expected 'fx' or 'fy'.");
}

double signed_double_area(const std::vector<fem::Node>& nodes) {
    const auto& n1 = nodes[0];
    const auto& n2 = nodes[1];
    const auto& n3 = nodes[2];
    return (n2.x - n1.x) * (n3.y - n1.y) - (n3.x - n1.x) * (n2.y - n1.y);
}

std::unordered_map<int, int> build_node_index(const std::vector<fem::Node>& nodes) {
    if (nodes.empty()) {
        throw std::invalid_argument("model must contain at least one node");
    }

    std::unordered_map<int, int> node_index_by_id;
    node_index_by_id.reserve(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const int node_id = nodes[i].id;
        if (node_index_by_id.find(node_id) != node_index_by_id.end()) {
            throw std::invalid_argument("duplicate node id detected: " + std::to_string(node_id));
        }
        node_index_by_id[node_id] = static_cast<int>(i);
    }

    return node_index_by_id;
}

std::unordered_map<int, const fem::Material*> build_material_index(
    const std::vector<fem::Material>& materials) {
    if (materials.empty()) {
        throw std::invalid_argument("model must contain at least one material");
    }

    std::unordered_map<int, const fem::Material*> material_by_id;
    material_by_id.reserve(materials.size());
    for (const auto& material : materials) {
        if (material.young_modulus <= 0.0) {
            throw std::invalid_argument("young_modulus must be positive");
        }
        if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
            throw std::invalid_argument("poisson_ratio must be in (-1.0, 0.5)");
        }

        if (material_by_id.find(material.id) != material_by_id.end()) {
            throw std::invalid_argument("duplicate material id detected: " + std::to_string(material.id));
        }
        material_by_id[material.id] = &material;
    }

    return material_by_id;
}

void validate_t3_geometry(
    const std::vector<fem::Node>& element_nodes,
    int element_id,
    double area_tolerance) {
    const double two_area = signed_double_area(element_nodes);
    const double area = 0.5 * two_area;

    if (std::abs(area) <= area_tolerance) {
        throw std::invalid_argument(
            "Element " + std::to_string(element_id) + " has near-zero area (" +
            std::to_string(area) + "). Degenerate T3 elements are not allowed.");
    }
    if (area < 0.0) {
        throw std::invalid_argument(
            "Element " + std::to_string(element_id) +
            " has clockwise node ordering (negative area). Use counter-clockwise connectivity for T3 elements.");
    }
}

std::vector<ResolvedElement> resolve_elements(
    const std::vector<fem::Element>& elements,
    const std::unordered_map<int, int>& node_index_by_id,
    const std::vector<fem::Node>& all_nodes,
    const std::unordered_map<int, const fem::Material*>& material_by_id,
    double area_tolerance) {
    if (elements.empty()) {
        throw std::invalid_argument("model must contain at least one element");
    }

    std::vector<ResolvedElement> resolved;
    resolved.reserve(elements.size());

    for (const auto& element : elements) {
        if (element.type != "T3") {
            throw std::invalid_argument(
                "Element " + std::to_string(element.id) + " has unsupported type '" + element.type +
                "'. solve_linear_static currently supports only T3 elements.");
        }
        if (element.connectivity.size() != 3) {
            throw std::invalid_argument(
                "Element " + std::to_string(element.id) +
                " must have exactly 3 node ids in connectivity for T3.");
        }

        std::vector<fem::Node> element_nodes;
        element_nodes.reserve(3);
        std::array<int, 3> node_indices{};
        for (std::size_t i = 0; i < 3; ++i) {
            const int node_id = element.connectivity[i];
            const auto node_it = node_index_by_id.find(node_id);
            if (node_it == node_index_by_id.end()) {
                throw std::invalid_argument(
                    "Element " + std::to_string(element.id) + " references unknown node id " +
                    std::to_string(node_id));
            }
            const int node_index = node_it->second;
            node_indices[i] = node_index;
            element_nodes.push_back(all_nodes[static_cast<std::size_t>(node_index)]);
        }

        validate_t3_geometry(element_nodes, element.id, area_tolerance);

        const auto material_it = material_by_id.find(element.material_id);
        if (material_it == material_by_id.end()) {
            throw std::invalid_argument(
                "Element " + std::to_string(element.id) + " references unknown material id " +
                std::to_string(element.material_id));
        }

        resolved.push_back(ResolvedElement{
            element.id,
            std::move(element_nodes),
            {
                2 * node_indices[0], 2 * node_indices[0] + 1,
                2 * node_indices[1], 2 * node_indices[1] + 1,
                2 * node_indices[2], 2 * node_indices[2] + 1,
            },
            material_it->second,
        });
    }

    return resolved;
}

std::vector<double> assemble_global_stiffness(
    int ndof,
    const std::vector<ResolvedElement>& resolved,
    double thickness) {
    std::vector<double> k_global(static_cast<std::size_t>(ndof) * static_cast<std::size_t>(ndof), 0.0);

    for (const auto& element : resolved) {
        const auto ke = fem::t3_stiffness_matrix(element.nodes, *element.material, thickness);
        if (ke.size() != 36) {
            throw std::runtime_error(
                "Element " + std::to_string(element.element_id) +
                " stiffness has invalid size. Expected 36 values.");
        }

        for (int i = 0; i < 6; ++i) {
            const int row = element.dof_indices[static_cast<std::size_t>(i)];
            for (int j = 0; j < 6; ++j) {
                const int col = element.dof_indices[static_cast<std::size_t>(j)];
                k_global[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
                         static_cast<std::size_t>(col)] +=
                    ke[static_cast<std::size_t>(i) * 6 + static_cast<std::size_t>(j)];
            }
        }
    }

    return k_global;
}

std::vector<double> assemble_global_force(
    int ndof,
    const std::vector<fem::Load>& loads,
    const std::unordered_map<int, int>& node_index_by_id) {
    std::vector<double> force(static_cast<std::size_t>(ndof), 0.0);

    for (const auto& load : loads) {
        const auto node_it = node_index_by_id.find(load.node_id);
        if (node_it == node_index_by_id.end()) {
            throw std::invalid_argument(
                "Load references unknown node id " + std::to_string(load.node_id));
        }

        const int dof = force_dof(load.dof, node_it->second);
        force[static_cast<std::size_t>(dof)] += load.value;
    }

    return force;
}

std::vector<std::pair<int, double>> collect_prescribed_dofs(
    const std::vector<fem::BoundaryCondition>& boundary_conditions,
    const std::unordered_map<int, int>& node_index_by_id) {
    std::unordered_map<int, double> prescribed;

    for (const auto& bc : boundary_conditions) {
        const auto node_it = node_index_by_id.find(bc.node_id);
        if (node_it == node_index_by_id.end()) {
            throw std::invalid_argument(
                "Boundary condition references unknown node id " + std::to_string(bc.node_id));
        }

        const int dof = displacement_dof(bc.dof, node_it->second);
        const auto existing = prescribed.find(dof);
        if (existing != prescribed.end()) {
            if (std::abs(existing->second - bc.value) > 1e-12) {
                throw std::invalid_argument(
                    "Conflicting displacement constraints on global dof " +
                    std::to_string(dof) + ": " + std::to_string(existing->second) +
                    " vs " + std::to_string(bc.value) + ".");
            }
        } else {
            prescribed[dof] = bc.value;
        }
    }

    std::vector<std::pair<int, double>> sorted_prescribed(prescribed.begin(), prescribed.end());
    std::sort(
        sorted_prescribed.begin(),
        sorted_prescribed.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    return sorted_prescribed;
}

void apply_displacement_bcs(
    int ndof,
    std::vector<double>& k_modified,
    std::vector<double>& f_modified,
    const std::vector<std::pair<int, double>>& prescribed) {
    for (const auto& [dof, value] : prescribed) {
        if (dof < 0 || dof >= ndof) {
            throw std::invalid_argument("Boundary condition DOF " + std::to_string(dof) + " is out of range");
        }

        for (int row = 0; row < ndof; ++row) {
            const double column_value =
                k_modified[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
                           static_cast<std::size_t>(dof)];
            f_modified[static_cast<std::size_t>(row)] -= column_value * value;
        }

        for (int col = 0; col < ndof; ++col) {
            k_modified[static_cast<std::size_t>(dof) * static_cast<std::size_t>(ndof) +
                       static_cast<std::size_t>(col)] = 0.0;
        }
        for (int row = 0; row < ndof; ++row) {
            k_modified[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
                       static_cast<std::size_t>(dof)] = 0.0;
        }

        k_modified[static_cast<std::size_t>(dof) * static_cast<std::size_t>(ndof) +
                   static_cast<std::size_t>(dof)] = 1.0;
        f_modified[static_cast<std::size_t>(dof)] = value;
    }
}

std::vector<double> solve_dense_linear_system(
    int ndof,
    const std::vector<double>& k_matrix,
    const std::vector<double>& rhs) {
    std::vector<double> a = k_matrix;
    std::vector<double> b = rhs;

    for (int pivot = 0; pivot < ndof; ++pivot) {
        int pivot_row = pivot;
        double pivot_abs = std::abs(
            a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(ndof) +
              static_cast<std::size_t>(pivot)]);
        for (int row = pivot + 1; row < ndof; ++row) {
            const double candidate = std::abs(
                a[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
                  static_cast<std::size_t>(pivot)]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if (pivot_abs <= kSingularTolerance) {
            throw std::runtime_error(
                "Failed to solve the linear system. Possible causes: insufficient displacement "
                "constraints (rigid body modes), invalid/degenerate element geometry, or "
                "inconsistent material parameters.");
        }

        if (pivot_row != pivot) {
            for (int col = 0; col < ndof; ++col) {
                std::swap(
                    a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(ndof) +
                      static_cast<std::size_t>(col)],
                    a[static_cast<std::size_t>(pivot_row) * static_cast<std::size_t>(ndof) +
                      static_cast<std::size_t>(col)]);
            }
            std::swap(b[static_cast<std::size_t>(pivot)], b[static_cast<std::size_t>(pivot_row)]);
        }

        const double diagonal =
            a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(ndof) +
              static_cast<std::size_t>(pivot)];

        for (int row = pivot + 1; row < ndof; ++row) {
            const double factor =
                a[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
                  static_cast<std::size_t>(pivot)] /
                diagonal;
            if (factor == 0.0) {
                continue;
            }

            a[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
              static_cast<std::size_t>(pivot)] = 0.0;
            for (int col = pivot + 1; col < ndof; ++col) {
                a[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
                  static_cast<std::size_t>(col)] -=
                    factor *
                    a[static_cast<std::size_t>(pivot) * static_cast<std::size_t>(ndof) +
                      static_cast<std::size_t>(col)];
            }
            b[static_cast<std::size_t>(row)] -= factor * b[static_cast<std::size_t>(pivot)];
        }
    }

    std::vector<double> x(static_cast<std::size_t>(ndof), 0.0);
    for (int row = ndof - 1; row >= 0; --row) {
        double sum = b[static_cast<std::size_t>(row)];
        for (int col = row + 1; col < ndof; ++col) {
            sum -= a[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
                     static_cast<std::size_t>(col)] *
                   x[static_cast<std::size_t>(col)];
        }

        const double diagonal =
            a[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
              static_cast<std::size_t>(row)];
        if (std::abs(diagonal) <= kSingularTolerance) {
            throw std::runtime_error(
                "Failed to solve the linear system. Possible causes: insufficient displacement "
                "constraints (rigid body modes), invalid/degenerate element geometry, or "
                "inconsistent material parameters.");
        }

        x[static_cast<std::size_t>(row)] = sum / diagonal;
    }

    for (const auto value : x) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Linear solver produced non-finite displacement values.");
        }
    }

    return x;
}

std::vector<double> compute_reactions(
    int ndof,
    const std::vector<double>& k_original,
    const std::vector<double>& displacement,
    const std::vector<double>& f_original) {
    std::vector<double> reactions(static_cast<std::size_t>(ndof), 0.0);
    for (int row = 0; row < ndof; ++row) {
        double value = 0.0;
        for (int col = 0; col < ndof; ++col) {
            value += k_original[static_cast<std::size_t>(row) * static_cast<std::size_t>(ndof) +
                                static_cast<std::size_t>(col)] *
                     displacement[static_cast<std::size_t>(col)];
        }
        reactions[static_cast<std::size_t>(row)] = value - f_original[static_cast<std::size_t>(row)];
    }

    return reactions;
}

std::vector<fem::ElementStrainStress> recover_element_results(
    const std::vector<ResolvedElement>& resolved,
    const std::vector<double>& displacement) {
    std::vector<fem::ElementStrainStress> element_results;
    element_results.reserve(resolved.size());

    for (const auto& element : resolved) {
        const auto b = fem::t3_b_matrix(element.nodes);
        const auto d = fem::elastic_matrix(*element.material);

        std::array<double, 6> ue{};
        for (int i = 0; i < 6; ++i) {
            ue[static_cast<std::size_t>(i)] =
                displacement[static_cast<std::size_t>(element.dof_indices[static_cast<std::size_t>(i)])];
        }

        std::vector<double> strain(3, 0.0);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 6; ++j) {
                strain[static_cast<std::size_t>(i)] +=
                    b[static_cast<std::size_t>(i) * 6 + static_cast<std::size_t>(j)] *
                    ue[static_cast<std::size_t>(j)];
            }
        }

        std::vector<double> stress(3, 0.0);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                stress[static_cast<std::size_t>(i)] +=
                    d[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(j)] *
                    strain[static_cast<std::size_t>(j)];
            }
        }

        element_results.push_back(fem::ElementStrainStress{
            element.element_id,
            std::move(strain),
            std::move(stress),
        });
    }

    return element_results;
}

std::vector<fem::NodeDisplacement> build_node_displacements(
    const std::vector<fem::Node>& nodes,
    const std::vector<double>& displacement) {
    std::vector<fem::NodeDisplacement> node_displacements;
    node_displacements.reserve(nodes.size());

    for (std::size_t i = 0; i < nodes.size(); ++i) {
        node_displacements.push_back(fem::NodeDisplacement{
            nodes[i].id,
            displacement[2 * i],
            displacement[2 * i + 1],
        });
    }

    return node_displacements;
}

fem::StaticSolveSummary build_summary(
    const std::vector<fem::Node>& nodes,
    const std::vector<fem::Element>& elements,
    const std::vector<double>& displacement) {
    double max_displacement = 0.0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const double ux = displacement[2 * i];
        const double uy = displacement[2 * i + 1];
        const double magnitude = std::sqrt(ux * ux + uy * uy);
        if (magnitude > max_displacement) {
            max_displacement = magnitude;
        }
    }

    return fem::StaticSolveSummary{
        nodes.size(),
        elements.size(),
        nodes.size() * 2,
        max_displacement,
    };
}

}  // namespace

namespace fem {

ModelSummary summarize_model(
    const std::vector<Node>& nodes,
    const std::vector<Element>& elements,
    const std::vector<Material>& materials) {
    return ModelSummary{
        nodes.size(),
        elements.size(),
        materials.size(),
    };
}

std::vector<double> solve_placeholder(std::size_t dof_count) {
    return std::vector<double>(dof_count, 0.0);
}

LinearStaticResult solve_linear_static(
    const std::vector<Node>& nodes,
    const std::vector<Element>& elements,
    const std::vector<Material>& materials,
    const std::vector<BoundaryCondition>& boundary_conditions,
    const std::vector<Load>& loads,
    double thickness,
    double area_tolerance) {
    if (thickness <= 0.0) {
        throw std::invalid_argument("thickness must be positive");
    }
    if (area_tolerance <= 0.0) {
        throw std::invalid_argument("area_tolerance must be positive");
    }

    const auto node_index_by_id = build_node_index(nodes);
    if (nodes.size() < 3) {
        throw std::invalid_argument("model must contain at least 3 nodes for T3 analysis");
    }

    const auto material_by_id = build_material_index(materials);
    const auto resolved =
        resolve_elements(elements, node_index_by_id, nodes, material_by_id, area_tolerance);

    const int ndof = static_cast<int>(nodes.size()) * 2;
    auto k_original = assemble_global_stiffness(ndof, resolved, thickness);
    auto f_original = assemble_global_force(ndof, loads, node_index_by_id);

    auto k_modified = k_original;
    auto f_modified = f_original;
    const auto prescribed = collect_prescribed_dofs(boundary_conditions, node_index_by_id);
    apply_displacement_bcs(ndof, k_modified, f_modified, prescribed);

    auto displacements = solve_dense_linear_system(ndof, k_modified, f_modified);
    auto reactions = compute_reactions(ndof, k_original, displacements, f_original);
    auto node_displacements = build_node_displacements(nodes, displacements);
    auto element_results = recover_element_results(resolved, displacements);
    auto summary = build_summary(nodes, elements, displacements);

    return LinearStaticResult{
        std::move(displacements),
        std::move(reactions),
        std::move(node_displacements),
        std::move(element_results),
        summary,
    };
}

}  // namespace fem

