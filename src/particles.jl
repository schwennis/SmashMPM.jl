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

    # c_p + |v_p| of the last G2P2G (or of the initial state); used for the CFL condition.
    # Stored per particle, so the P2G needs no atomic max on the grid.
    wave_speed::T = zero(mass)
end


# ---------------------------------------------------------------------------- #
#                           Maximum wave speed (CFL)                           #
# ---------------------------------------------------------------------------- #
@inline _max_wavespeed(ws::AbstractArray{T}) where {T} = maximum(ws; init=zero(T))

"""
    max_wavespeed(particle_sets)

Largest particle wave speed c_p + |v_p| of all particle sets (0 if there are no
particles). Plain reduction over the particle arrays, used for the CFL condition.
Padding particles of an AoSoA set have wave speed 0 and do not matter.
"""
max_wavespeed(particle_sets::Tuple) = maximum(ps -> _max_wavespeed(ps.particles.wave_speed), particle_sets)


include("particles/SoA_particles.jl")
include("particles/AoSoA_particles.jl")