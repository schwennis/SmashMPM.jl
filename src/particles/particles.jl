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


# ---------------------------------------------------------------------------- #
#                               SoA Particle Set                               #
# ---------------------------------------------------------------------------- #
mutable struct SoAParticleSet{MatType<:AbstractMaterial, SA<:StructArray}<:AbstractParticleSet
    particles::SA              # StructArray{Particle{T,MaterialState}}
    particles_buffer::SA       # Buffer for fast swapping of points (StructArray{Particle{T,MaterialState}})
    material::MatType
end



function _allocate_soa(backend, particle_vector::Vector{Particle{T,MS}}) where {T, MS}
    n = length(particle_vector)

    # Fill on CPU
    ids = Vector{UInt32}(undef, n)
    pos_x = Vector{T}(undef, n); pos_y = Vector{T}(undef, n); pos_z = Vector{T}(undef, n)
    mass   = Vector{T}(undef, n)
    vol    = Vector{T}(undef, n)
    F      = Vector{SMatrix{3,3,T,9}}(undef, n)
    mstate = Vector{MS}(undef, n)

    @inbounds for i in eachindex(particle_vector)
        p = particle_vector[i]
        ids[i] = p.id
        pos_x[i] = p.pos[1]; pos_y[i] = p.pos[2]; pos_z[i] = p.pos[3]
        mass[i]   = p.mass
        vol[i]    = p.initial_volume
        F[i]      = p.F
        mstate[i] = p.mat_state
    end

    # transfer to backend; Manual unwrap to prevent type instabilities
    pos = StructArray{SVector{3,T}}((
        x = _to_backend(backend, pos_x),
        y = _to_backend(backend, pos_y),
        z = _to_backend(backend, pos_z)
    ))
    return StructArray{Particle{T,MS}}((
        id              = _to_backend(backend, ids),
        pos             = pos,
        mass            = _to_backend(backend, mass),
        initial_volume  = _to_backend(backend, vol),
        F               = _to_backend(backend, F),
        mat_state       = _to_backend(backend, mstate)
    ))
end

function SoAParticleSet(particle_vector::Vector{Particle{T,MS}},
                         material::MaterialType,
                         backend=CPU()) where {T, MS, MaterialType<:AbstractMaterial}
    SoA        = _allocate_soa(backend, particle_vector)
    SoA_buffer = _allocate_soa(backend, particle_vector)
    return SoAParticleSet{MaterialType, typeof(SoA)}(SoA, SoA_buffer, material)
end
