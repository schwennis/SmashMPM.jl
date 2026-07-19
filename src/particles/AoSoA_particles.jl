# ---------------------------------------------------------------------------- #
#                              AoSoA Particle Set                              #
# ---------------------------------------------------------------------------- #
mutable struct AoSoAParticleSet{BS, MatType<:AbstractMaterial, SA<:StructArray}<:AbstractParticleSet
    particles::SA              # StructArray{Particle{T,MaterialState}}
    particles_buffer::SA       # Buffer for fast swapping of points (StructArray{Particle{T,MaterialState}})
    material::MatType
    active_blocks::Int
    max_blocks::Int
end


function _allocate_aosoa(backend, particle_vector::Vector{Particle{T,MS}}, block_size::Int, max_blocks::Int) where {T, MS}
    n_padded = max_blocks * block_size
    n_original = length(particle_vector)
    
    # ... Vektoren in n_padded Länge erstellen ...
    ids = Vector{UInt32}(undef, n_padded)
    pos_x = Vector{T}(undef, n_padded); pos_y = Vector{T}(undef, n_padded); pos_z = Vector{T}(undef, n_padded)
    mass   = zeros(T, n_padded)  # Initialisiere mit 0 für Dummy-Particles
    vol    = Vector{T}(undef, n_padded)
    F      = Vector{SMatrix{3,3,T,9}}(undef, n_padded)
    mstate = Vector{MS}(undef, n_padded)    

    
    # ... particle_vector[1:n_original] reinkopieren ...
    @inbounds for i in 1:n_original
        p = particle_vector[i]
        ids[i] = p.id
        pos_x[i] = p.pos[1]; pos_y[i] = p.pos[2]; pos_z[i] = p.pos[3]
        mass[i]   = p.mass
        vol[i]    = p.initial_volume
        F[i]      = p.F
        mstate[i] = p.mat_state
    end

    # ... Rest (n_original+1 bis n_padded) mit 0-Masse (Dummy) auffüllen ...
    dummy_mat_state = length(particle_vector) > 0 ? particle_vector[1].mat_state : NoMaterialState()
    @inbounds for i in (n_original+1):n_padded
        ids[i] = UInt32(0)
        pos_x[i] = zero(T); pos_y[i] = zero(T); pos_z[i] = zero(T)
        mass[i]   = zero(T)
        vol[i]    = zero(T)
        F[i]      = one(SMatrix{3,3,T,9})
        mstate[i] = deepcopy(dummy_mat_state)
    end

    # ... Reshape & KernelAbstractions Transfer aufs Backend (wie du es schon hast) ...
    pos = StructArray{SVector{3,T}}((
        x = reshape(_to_backend(backend, pos_x), block_size, max_blocks),
        y = reshape(_to_backend(backend, pos_y), block_size, max_blocks),
        z = reshape(_to_backend(backend, pos_z), block_size, max_blocks)
    ))

    SA = StructArray{Particle{T,MS}}((
        id              = reshape(_to_backend(backend, ids), block_size, max_blocks),
        pos             = pos,
        mass            = reshape(_to_backend(backend, mass), block_size, max_blocks),
        initial_volume  = reshape(_to_backend(backend, vol), block_size, max_blocks),
        F               = reshape(_to_backend(backend, F), block_size, max_blocks),
        mat_state       = reshape(_to_backend(backend, mstate), block_size, max_blocks)
    ))

    # Rückgabe des Arrays
end


function AoSoAParticleSet(particle_vector::Vector{Particle{T,MS}}, material::MaterialType, backend, ::Val{block_size}, overprovision_factor::Float64=1.0) where {T, MS, MaterialType<:AbstractMaterial, block_size}
    n_original = length(particle_vector)
    active_blocks = ceil(Int, n_original / block_size)
    max_blocks = ceil(Int, active_blocks * overprovision_factor)
    
    AoSoA = _allocate_aosoa(backend, particle_vector, block_size, max_blocks)
    AoSoA_buffer = _allocate_aosoa(backend, particle_vector, block_size, max_blocks)
    
    return AoSoAParticleSet{block_size, MaterialType, typeof(AoSoA)}(
        AoSoA, AoSoA_buffer, material, block_size, active_blocks, max_blocks
    )
end