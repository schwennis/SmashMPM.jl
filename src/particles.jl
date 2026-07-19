abstract type AbstractParticleSet end

struct Particle{T, MS<:AbstractMaterialState}
    id::UInt32

    pos::SVector{3, T}

    mass::T
    initial_volume::T

    F::SMatrix{3, 3, T, 9}

    mat_state::MS
end


function Particle(id::Int, position::SVector{3, T}, mass::T, initial_volume::T, mat_state::MS) where {T, MS<:AbstractMaterialState}
    return Particle{T, MS}(UInt32(id), position, mass, initial_volume, one(SMatrix{3, 3, T, 9}), mat_state)
end


include("particles/SoA_particles.jl")
include("particles/AoSoA_particles.jl")