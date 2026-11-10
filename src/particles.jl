abstract type AbstractParticleSet end

@kwdef struct Particle{T, MS<:AbstractMaterialState}
    id::UInt32

    pos::SVector{3, T}
    vel::SVector{3, T} = zero(pos)

    mass::T
    V0::T

    F::SMatrix{3, 3, T, 9} = one(SMatrix{3, 3, eltype(pos), 9})
    C::SMatrix{3, 3, T, 9} = zero(F)

    mat_state::MS
end


include("particles/SoA_particles.jl")
include("particles/AoSoA_particles.jl")