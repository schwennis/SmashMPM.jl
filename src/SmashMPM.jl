module SmashMPM


using StructArrays
using StaticArrays
using LinearAlgebra

using KernelAbstractions
using Adapt
using Atomix: @atomic

using NearestNeighbors


# -------------------------------- Core Folder ------------------------------- #
include("core/abstract_types.jl")
export NoMaterialCache, NoBoundaryCondition, NoExternalForce
include("core/model.jl")
export Model


# ------------------------------ Particle Folder ----------------------------- #
include("particles/particle.jl")
export Particle
include("particles/SoA_particles.jl")
export SoAMaterialGroup


# -------------------------------- Grid Folder ------------------------------- #
include("grid/grid_node.jl")
export GridNode
include("grid/dense_grid.jl")
export DenseGrid


# --------------------------- Shape Function Folder -------------------------- #
include("shapefunctions/quadratic_spline.jl")
export shapefunction, QuadraticSpline
include("shapefunctions/shapefunction_utils.jl")


# ----------------------------- Materials Folder ----------------------------- #
include("materials/NeoHookean.jl")
export NeoHookean, material_model


# -------------------------- External Forces Folder -------------------------- #
include("external_forces/const_gravity.jl")
export ConstantGravity
include("external_forces/self_gravity_FFT.jl")
export SelfGravityFFT
include("external_forces/no_external_force.jl")


# ------------------------ Boundary Conditions Folder ------------------------ #
include("boundary_conditions/no_boundary.jl")
export NoBoundaryCondition
include("boundary_conditions/no_slip.jl")
export NoSlipBoundary


# ------------------------------- Solver Folder ------------------------------ #
include("solver/courant_timestep.jl")
export courant_timestep
include("solver/g2p2g_kernel.jl")
export g2p2g_kernel
include("solver/step.jl")
export step!


# ------------------------------- Setup Folder ------------------------------ #
include("setup/autogrid.jl")
export autogrid_parameters!
include("setup/initialize_grid.jl")
export initial_p2g!
include("setup/initialize_particles.jl")
export add_material_group!


end