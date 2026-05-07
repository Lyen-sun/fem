#include "fem_core/elements.hpp"

#include <array>
#include <stdexcept>

namespace {

using Matrix3 = std::array<std::array<double, 3>, 3>;
using Matrix3x6 = std::array<std::array<double, 6>, 3>;
using Vector3 = std::array<double, 3>;

void validate_material(const fem::Material& material) {
    if (material.young_modulus <= 0.0) {
        throw std::invalid_argument("young_modulus must be positive");
    }
    if (material.poisson_ratio <= -1.0 || material.poisson_ratio >= 0.5) {
        throw std::invalid_argument("poisson_ratio must be in (-1.0, 0.5)");
    }
}

void validate_t3_nodes(const std::vector<fem::Node>& nodes) {
    if (nodes.size() != 3) {
        throw std::invalid_argument("T3 element requires exactly 3 nodes");
    }
}

Matrix3 make_elastic_matrix(const fem::Material& material) {
    validate_material(material);

    Matrix3 matrix{};
    const double e = material.young_modulus;
    const double nu = material.poisson_ratio;

    if (material.plane_stress) {
        const double factor = e / (1.0 - nu * nu);
        matrix[0][0] = factor;
        matrix[0][1] = factor * nu;
        matrix[1][0] = factor * nu;
        matrix[1][1] = factor;
        matrix[2][2] = factor * (1.0 - nu) * 0.5;
        return matrix;
    }

    const double factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
    matrix[0][0] = factor * (1.0 - nu);
    matrix[0][1] = factor * nu;
    matrix[1][0] = factor * nu;
    matrix[1][1] = factor * (1.0 - nu);
    matrix[2][2] = factor * (1.0 - 2.0 * nu) * 0.5;
    return matrix;
}

void fill_t3_coefficients(
    const std::vector<fem::Node>& nodes,
    double& two_area,
    Vector3& beta,
    Vector3& gamma) {
    validate_t3_nodes(nodes);

    const auto& n1 = nodes[0];
    const auto& n2 = nodes[1];
    const auto& n3 = nodes[2];

    two_area =
        (n2.x - n1.x) * (n3.y - n1.y) - (n3.x - n1.x) * (n2.y - n1.y);
    if (two_area <= 0.0) {
        throw std::invalid_argument("T3 area must be positive for counter-clockwise node ordering");
    }

    beta[0] = n2.y - n3.y;
    beta[1] = n3.y - n1.y;
    beta[2] = n1.y - n2.y;

    gamma[0] = n3.x - n2.x;
    gamma[1] = n1.x - n3.x;
    gamma[2] = n2.x - n1.x;
}

}  // namespace

namespace fem {

std::vector<double> elastic_matrix(const Material& material) {
    const auto matrix = make_elastic_matrix(material);
    return {
        matrix[0][0], matrix[0][1], matrix[0][2],
        matrix[1][0], matrix[1][1], matrix[1][2],
        matrix[2][0], matrix[2][1], matrix[2][2],
    };
}

double t3_area(const std::vector<Node>& nodes) {
    double two_area = 0.0;
    Vector3 beta{};
    Vector3 gamma{};
    fill_t3_coefficients(nodes, two_area, beta, gamma);
    return 0.5 * two_area;
}

std::vector<double> t3_b_matrix(const std::vector<Node>& nodes) {
    double two_area = 0.0;
    Vector3 beta{};
    Vector3 gamma{};
    fill_t3_coefficients(nodes, two_area, beta, gamma);

    const double inv_two_area = 1.0 / two_area;

    return {
        beta[0] * inv_two_area, 0.0, beta[1] * inv_two_area, 0.0, beta[2] * inv_two_area, 0.0,
        0.0, gamma[0] * inv_two_area, 0.0, gamma[1] * inv_two_area, 0.0, gamma[2] * inv_two_area,
        gamma[0] * inv_two_area, beta[0] * inv_two_area, gamma[1] * inv_two_area, beta[1] * inv_two_area, gamma[2] * inv_two_area, beta[2] * inv_two_area,
    };
}

std::vector<double> t3_stiffness_matrix(
    const std::vector<Node>& nodes,
    const Material& material,
    double thickness) {
    if (thickness <= 0.0) {
        throw std::invalid_argument("thickness must be positive");
    }

    const auto d = make_elastic_matrix(material);

    double two_area = 0.0;
    Vector3 beta{};
    Vector3 gamma{};
    fill_t3_coefficients(nodes, two_area, beta, gamma);

    const double inv_two_area = 1.0 / two_area;
    const double area = 0.5 * two_area;
    Matrix3x6 b{{
        {beta[0] * inv_two_area, 0.0, beta[1] * inv_two_area, 0.0, beta[2] * inv_two_area, 0.0},
        {0.0, gamma[0] * inv_two_area, 0.0, gamma[1] * inv_two_area, 0.0, gamma[2] * inv_two_area},
        {gamma[0] * inv_two_area, beta[0] * inv_two_area, gamma[1] * inv_two_area, beta[1] * inv_two_area, gamma[2] * inv_two_area, beta[2] * inv_two_area},
    }};

    Matrix3x6 db{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 3; ++k) {
                db[i][j] += d[i][k] * b[k][j];
            }
        }
    }

    std::vector<double> ke(36, 0.0);
    const double scale = thickness * area;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            double value = 0.0;
            for (int k = 0; k < 3; ++k) {
                value += b[k][i] * db[k][j];
            }
            ke[static_cast<std::size_t>(i) * 6 + static_cast<std::size_t>(j)] = scale * value;
        }
    }

    return ke;
}

}  // namespace fem
