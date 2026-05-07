cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(fem_core
    src/bindings.cpp
    src/elements.cpp
    src/model.cpp
    src/solver.cpp
)

target_include_directories(fem_core PRIVATE include)
