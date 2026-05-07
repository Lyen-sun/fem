#include "fem_core/elements.hpp"
#include "fem_core/model.hpp"
#include "fem_core/solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(fem_core, m) {
    m.doc() = "Core FEM bindings";

    py::class_<fem::Node>(m, "Node")
        .def(py::init<>())
        .def_readwrite("id", &fem::Node::id)
        .def_readwrite("x", &fem::Node::x)
        .def_readwrite("y", &fem::Node::y);

    py::class_<fem::Material>(m, "Material")
        .def(py::init<>())
        .def_readwrite("id", &fem::Material::id)
        .def_readwrite("young_modulus", &fem::Material::young_modulus)
        .def_readwrite("poisson_ratio", &fem::Material::poisson_ratio)
        .def_readwrite("plane_stress", &fem::Material::plane_stress);

    py::class_<fem::Element>(m, "Element")
        .def(py::init<>())
        .def_readwrite("id", &fem::Element::id)
        .def_readwrite("type", &fem::Element::type)
        .def_readwrite("connectivity", &fem::Element::connectivity)
        .def_readwrite("material_id", &fem::Element::material_id);

    py::class_<fem::BoundaryCondition>(m, "BoundaryCondition")
        .def(py::init<>())
        .def_readwrite("node_id", &fem::BoundaryCondition::node_id)
        .def_readwrite("dof", &fem::BoundaryCondition::dof)
        .def_readwrite("value", &fem::BoundaryCondition::value);

    py::class_<fem::Load>(m, "Load")
        .def(py::init<>())
        .def_readwrite("node_id", &fem::Load::node_id)
        .def_readwrite("dof", &fem::Load::dof)
        .def_readwrite("value", &fem::Load::value);

    py::class_<fem::ModelSummary>(m, "ModelSummary")
        .def(py::init<>())
        .def_readonly("node_count", &fem::ModelSummary::node_count)
        .def_readonly("element_count", &fem::ModelSummary::element_count)
        .def_readonly("material_count", &fem::ModelSummary::material_count);

    py::class_<fem::NodeDisplacement>(m, "NodeDisplacement")
        .def(py::init<>())
        .def_readonly("node_id", &fem::NodeDisplacement::node_id)
        .def_readonly("ux", &fem::NodeDisplacement::ux)
        .def_readonly("uy", &fem::NodeDisplacement::uy);

    py::class_<fem::ElementStrainStress>(m, "ElementStrainStress")
        .def(py::init<>())
        .def_readonly("element_id", &fem::ElementStrainStress::element_id)
        .def_readonly("strain", &fem::ElementStrainStress::strain)
        .def_readonly("stress", &fem::ElementStrainStress::stress);

    py::class_<fem::StaticSolveSummary>(m, "StaticSolveSummary")
        .def(py::init<>())
        .def_readonly("node_count", &fem::StaticSolveSummary::node_count)
        .def_readonly("element_count", &fem::StaticSolveSummary::element_count)
        .def_readonly("total_dof", &fem::StaticSolveSummary::total_dof)
        .def_readonly("max_displacement", &fem::StaticSolveSummary::max_displacement);

    py::class_<fem::LinearStaticResult>(m, "LinearStaticResult")
        .def(py::init<>())
        .def_readonly("displacements", &fem::LinearStaticResult::displacements)
        .def_readonly("reactions", &fem::LinearStaticResult::reactions)
        .def_readonly("node_displacements", &fem::LinearStaticResult::node_displacements)
        .def_readonly("element_results", &fem::LinearStaticResult::element_results)
        .def_readonly("summary", &fem::LinearStaticResult::summary);

    m.attr("StaticSolveResult") = m.attr("LinearStaticResult");

    m.def("summarize_model", &fem::summarize_model);
    m.def("solve_placeholder", &fem::solve_placeholder);
    m.def("elastic_matrix", &fem::elastic_matrix);
    m.def("t3_area", &fem::t3_area);
    m.def("t3_b_matrix", &fem::t3_b_matrix);
    m.def(
        "t3_stiffness_matrix",
        &fem::t3_stiffness_matrix,
        py::arg("nodes"),
        py::arg("material"),
        py::arg("thickness") = 1.0);
    m.def(
        "solve_linear_static",
        &fem::solve_linear_static,
        py::arg("nodes"),
        py::arg("elements"),
        py::arg("materials"),
        py::arg("boundary_conditions"),
        py::arg("loads"),
        py::arg("thickness") = 1.0,
        py::arg("area_tolerance") = 1e-14);
}
