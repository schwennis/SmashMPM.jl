module SmashMPM

using StructArrays
using StaticArrays
using LinearAlgebra

using KernelAbstractions
using Adapt
using Atomix: @atomic

using JLD2
using HDF5
using Printf

include("helpers.jl")

include("materials.jl")
export AbstractMaterial, AbstractMaterialState, NoMaterialState     # Abstract types
export material_model, get_soundspeed, get_initial_material_state   # Material model interface
export NeoHookean, LinearElastic                                    # hyperelastic materials
export NoMaterialModel                                              # Debugging / Fallback
export SolidMaterial, SolidMaterialState                            # Solid materials
export MurnaghanEOS, MurnaghanState                                 # Murnaghan EOS
export TillotsonEOS, TillotsonState                                 # Tillotson EOS
export HypoElasticStrengthModel, HypoElasticStrengthModelState      # Hypoelastic strength model
export HyperElasticStrengthModel, HyperElasticStrengthModelState    # Hyperelastic strength model
# Material configs
export Basalt, Iron

include("shapefunctions.jl")
export QuadraticSpline
export shapefunction


include("particles.jl")
export AbstractParticleSet, Particle, SoAParticleSet

include("grid.jl")
export AbstractGrid, DenseGrid, GridNode
export grid_reset!

include("boundary_conditions.jl")
export AbstractBoundaryCondition, NoBoundaryCondition, NoSlipBoundary
export apply_boundary_condition!

include("external_forces.jl")
export AbstractExternalForce, NoExternalForce, ConstantGravity, RadialInvSquareForceField
export apply_external_forces!

include("setup/geometry.jl")
export AbstractShape, Sphere, Cylinder, RectangularPrism
export generate_particles

include("mpm_model.jl")
export MPMModel

include("setup/initial_p2g.jl")

include("export/extract_velocities.jl")
include("export/exporter.jl")
include("export/extract_particle_fields.jl")
export AbstractExporter, HDF5Exporter, VTKExporter, write_output

include("setup/build_sim.jl")
export AbstractBody, Body, SimulationSetup, build_mpm_model

include("solver.jl")
export g2p2g!, courant_timestep

include("solver/solve.jl")
export step!, solve!
end