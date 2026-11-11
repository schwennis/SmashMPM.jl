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

    # Fill on CPU (StructArray) and then transfer to backend
    ids = Vector{UInt32}(undef, n)
    pos_x = Vector{T}(undef, n); pos_y = Vector{T}(undef, n); pos_z = Vector{T}(undef, n)
    vel_x = Vector{T}(undef, n); vel_y = Vector{T}(undef, n); vel_z = Vector{T}(undef, n)
    mass   = Vector{T}(undef, n)
    vol    = Vector{T}(undef, n)
    F      = Vector{SMatrix{3,3,T,9}}(undef, n)
    C      = Vector{SMatrix{3,3,T,9}}(undef, n)
    mstate = Vector{MS}(undef, n)
    ws     = Vector{T}(undef, n)

    @inbounds for i in eachindex(particle_vector)
        p = particle_vector[i]
        ids[i] = p.id
        pos_x[i] = p.pos[1]; pos_y[i] = p.pos[2]; pos_z[i] = p.pos[3]
        vel_x[i] = p.vel[1]; vel_y[i] = p.vel[2]; vel_z[i] = p.vel[3]
        mass[i]     = p.mass
        vol[i]      = p.V0
        F[i]        = p.F
        C[i]        = p.C
        mstate[i]   = p.mat_state
        ws[i]       = p.wave_speed
    end

    # transfer to backend; Manual unwrap to prevent type instabilities
    pos = StructArray{SVector{3,T}}((
        x = _to_backend(backend, pos_x),
        y = _to_backend(backend, pos_y),
        z = _to_backend(backend, pos_z)
    ))
    vel = StructArray{SVector{3,T}}((
        x = _to_backend(backend, vel_x),
        y = _to_backend(backend, vel_y),
        z = _to_backend(backend, vel_z)
    ))
    return StructArray{Particle{T,MS}}((
        id              = _to_backend(backend, ids),
        pos             = pos,
        vel             = vel,
        mass            = _to_backend(backend, mass),
        V0  = _to_backend(backend, vol),
        F               = _to_backend(backend, F),
        C               = _to_backend(backend, C),
        mat_state       = _to_backend(backend, mstate),
        wave_speed      = _to_backend(backend, ws)
    ))
end

function SoAParticleSet(particle_vector::Vector{Particle{T,MS}},
                         material::MaterialType,
                         backend=CPU()) where {T, MS, MaterialType<:AbstractMaterial}
    SoA        = _allocate_soa(backend, particle_vector)
    SoA_buffer = _allocate_soa(backend, particle_vector)
    return SoAParticleSet{MaterialType, typeof(SoA)}(SoA, SoA_buffer, material)
end


function get_particle_index(soa::SoAParticleSet, p_idx::Int)
    return p_idx
end