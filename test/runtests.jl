# ---------------------------------------------------------------------------- #
#                              SmashMPM test suite                             #
# ---------------------------------------------------------------------------- #
# Run from the repository root:
#   julia -t auto --project tests/runtests.jl
# Only some of the files (names as in TEST_FILES below):
#   julia -t auto --project tests/runtests.jl materials simulation
#
# The suite checks two things:
#   1. physical correctness: conservation laws, energy budgets, analytic
#      solutions and thermodynamic consistency of the material models
#   2. type stability of everything that runs inside the kernels (also Float32,
#      i.e. GPU precision) and allocation-free hot paths
#
# Loops are reduced to ONE assertion per physical statement (max error over all
# samples), so a failure names the broken property instead of producing
# hundreds of red lines.
# ---------------------------------------------------------------------------- #
include("../src/SmashMPM.jl")
using .SmashMPM

using Test
using JET
using LinearAlgebra
using Logging
using Random
using StaticArrays
using StructArrays
using KernelAbstractions

include("helpers.jl")

# Optional selection of test files: `julia tests/runtests.jl materials grids`
const TEST_FILES = [
    "type_stability",
    "shapefunctions",
    "materials",
    "grids",
    "simulation",
    "export",
]
const SELECTED = isempty(ARGS) ? TEST_FILES : filter(f -> f in ARGS, TEST_FILES)

@testset "SmashMPM" begin
    for name in SELECTED
        @testset "$name" begin
            include("test_$(name).jl")
        end
    end
end
