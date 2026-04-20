module SmashMPM


using StructArrays
using StaticArrays
using LinearAlgebra

using KernelAbstractions
using adapt

include("core/abstract_types.jl")
include("core/model.jl")
export Model

include("particles/particle.jl")
export Particle

include("particles/SoA_particles.jl")
export SoAMaterialGroup

include("grid/grid_node.jl")
export GridNode

include("grid/dense_grid.jl")
export DenseGrid

end