module SmashMPM

using StructArrays
using StaticArrays
using LinearAlgebra

using KernelAbstractions
using Adapt
using Atomix: @atomic

using JLD2
using Printf
using WriteVTK

include("helpers.jl")

include("materials.jl")
export AbstractMaterial, AbstractMaterialState, NoMaterialState
export material_model, get_soundspeed, get_initial_material_state
export NeoHookean, LinearElastic, NoMaterialModel

include("shapefunctions.jl")
export QuadraticSpline
export shapefunction


include("particles.jl")
export AbstractParticleSet, Particle, SoAParticleSet

include("grid.jl")
export AbstractGrid, DenseGrid, GridNode

include("boundary_conditions.jl")
export AbstractBoundaryCondition, NoBoundaryCondition, NoSlipBoundary
export apply_boundary_condition!

include("external_forces.jl")
export AbstractExternalForce, NoExternalForce, ConstantGravity, RadialInvSquareForceField
export apply_external_forces!

include("setup/geometry.jl")
export AbstractShape, Sphere, Cylinder
export generate_particles

include("mpm_model.jl")
export MPMModel

include("setup/initial_p2g.jl")

include("setup/build_sim.jl")
export AbstractBody, Body, SimulationSetup, build_mpm_model

include("export/exporter.jl")
include("export/extract_velocities.jl")

include("solver.jl")
export g2p2g!, courant_timestep
end