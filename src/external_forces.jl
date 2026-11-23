# ============================================================================ #
#                               EXTERNAL FORCES                                #
# ============================================================================ #

abstract type AbstractExternalForce end

"""Position of the global node (i, j, k)."""
@inline function node_position(origin::SVector{3, T}, inv_dx::T, i, j, k) where {T}
    dx = one(T) / inv_dx
    return SVector{3, T}(origin[1] + dx * (i - 1),
                         origin[2] + dx * (j - 1),
                         origin[3] + dx * (k - 1))
end


# ---------------------------------------------------------------------------- #
#                               No External Force                              #
# ---------------------------------------------------------------------------- #
struct NoExternalForce <: AbstractExternalForce end

@inline function apply_external_forces!(::NoExternalForce, grid::AbstractGrid, dt::Real)
    return nothing
end


# ---------------------------------------------------------------------------- #
#                               Constant Gravity                               #
# ---------------------------------------------------------------------------- #
@kwdef struct ConstantGravity{T} <: AbstractExternalForce
    g::SVector{3, T}
end

function apply_external_forces!(force::ConstantGravity, grid::DenseGrid{T, S}, dt::Real) where {T, S}
    state = grid.state_write

    state.force.x .+= state.mass .* force.g[1]
    state.force.y .+= state.mass .* force.g[2]
    state.force.z .+= state.mass .* force.g[3]
end

function apply_external_forces!(force::ConstantGravity, grid::SparseGrid{T}, dt::Real) where {T}
    grid.n_active_write == 0 && return nothing
    state = grid.state_write
    mass  = active_view(grid, state.mass)


    active_view(grid, state.force.x) .+= mass .* force.g[1]
    active_view(grid, state.force.y) .+= mass .* force.g[2]
    active_view(grid, state.force.z) .+= mass .* force.g[3]
    return nothing
end


# ---------------------------------------------------------------------------- #
#                              Radial Force Field                              #
# ---------------------------------------------------------------------------- #
@kwdef struct RadialInvSquareForceField{T} <: AbstractExternalForce
    F_0::T
    center::SVector{3, T}
end

"""Force per unit mass of the radial inverse square field at `pos`."""
@inline function radial_inv_square_force(pos::SVector{3, T}, center, F_0) where {T}
    r_vec = pos .- center
    r = norm(r_vec) + eps(T)
    return T(F_0) * (r_vec ./ (r^3))
end

# --- DenseGrid ---------------------------------------------------------------- #
@kernel function radial_inv_square_force_kernel!(force_x, force_y, force_z, mass, origin, inv_dx, center, F_0)
    # Direkte 3D-Indexierung über GPU-Register (vermeidet div/mod)
    i, j, k = @index(Global, NTuple)

    m = mass[i, j, k]
    T = eltype(mass)

    # Nur auswerten, wenn am Knoten tatsächlich Masse anliegt
    if m > eps(T)
        pos = node_position(origin, inv_dx, i, j, k)
        force_vec = radial_inv_square_force(pos, center, F_0)

        force_x[i, j, k] += m * force_vec[1]
        force_y[i, j, k] += m * force_vec[2]
        force_z[i, j, k] += m * force_vec[3]
    end
end

function apply_external_forces!(force::RadialInvSquareForceField, grid::DenseGrid{T, S}, dt::Real) where {T, S}
    state = grid.state_write

    backend = KernelAbstractions.get_backend(state.mass)
    kernel = radial_inv_square_force_kernel!(backend)

    kernel(
        state.force.x,
        state.force.y,
        state.force.z,
        state.mass,
        grid.origin,
        grid.inv_dx,
        force.center,
        force.F_0,
        ndrange=size(state.mass) # 3D-Range
    )

    KernelAbstractions.synchronize(backend)
end

# --- SparseGrid --------------------------------------------------------------- #
@kernel function sparse_radial_inv_square_force_kernel!(force_x, force_y, force_z, mass, tile_keys,
                                                        origin, inv_dx, center, F_0)
    l, t = @index(Global, NTuple)

    m = mass[l, t]
    T = eltype(mass)

    if m > eps(T)
        i, j, k = node_index(tile_keys[t], l)
        pos = node_position(origin, inv_dx, i, j, k)
        force_vec = radial_inv_square_force(pos, center, F_0)

        force_x[l, t] += m * force_vec[1]
        force_y[l, t] += m * force_vec[2]
        force_z[l, t] += m * force_vec[3]
    end
end

function apply_external_forces!(force::RadialInvSquareForceField, grid::SparseGrid{T}, dt::Real) where {T}
    n = grid.n_active_write
    n == 0 && return nothing
    state = grid.state_write

    backend = KernelAbstractions.get_backend(state.mass)
    kernel = sparse_radial_inv_square_force_kernel!(backend)

    kernel(
        state.force.x, state.force.y, state.force.z,
        state.mass, grid.map_write.tile_keys,
        grid.origin, grid.inv_dx, force.center, force.F_0;
        ndrange=(TILE_VOLUME, n)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end


# ---------------------------------------------------------------------------- #
#                              Vector Field Force                              #
# ---------------------------------------------------------------------------- #
# `force_field[i, j, k]` is the force per unit mass at the dense node (i, j, k).
# On a SparseGrid this requires a BoxDomain; dense index = global index - lo + 1.
# (A field over an unbounded domain cannot be stored as an array.)
@kwdef struct VectorFieldForce{A} <: AbstractExternalForce
    force_field::A
end

# --- DenseGrid ---------------------------------------------------------------- #
@kernel function vector_field_force_kernel!(force_x, force_y, force_z, mass, force_field)
    i, j, k = @index(Global, NTuple)

    m = mass[i, j, k]
    if m > eps(eltype(mass))
        f_vec = force_field[i, j, k]
        force_x[i, j, k] += m * f_vec[1]
        force_y[i, j, k] += m * f_vec[2]
        force_z[i, j, k] += m * f_vec[3]
    end
end

function apply_external_forces!(force::VectorFieldForce, grid::DenseGrid{T, S}, dt::Real) where {T, S}
    state = grid.state_write
    force_field = force.force_field

    @assert size(force_field) == size(state.mass) "Force field dimensions must match grid dimensions."

    backend = KernelAbstractions.get_backend(state.mass)

    # Sicherstellen, dass das Feld auf dem Device liegt
    dev_force_field = _to_backend(backend, force_field)

    kernel = vector_field_force_kernel!(backend)
    kernel(
        state.force.x,
        state.force.y,
        state.force.z,
        state.mass,
        dev_force_field,
        ndrange=size(state.mass)
    )

    KernelAbstractions.synchronize(backend)
end

# --- SparseGrid --------------------------------------------------------------- #
@kernel function sparse_vector_field_force_kernel!(force_x, force_y, force_z, mass, tile_keys,
                                                   force_field, lo)
    l, t = @index(Global, NTuple)

    m = mass[l, t]
    if m > eps(eltype(mass))
        i, j, k = node_index(tile_keys[t], l)
        a, b, c = i - lo[1] + 1, j - lo[2] + 1, k - lo[3] + 1
        if checkbounds(Bool, force_field, a, b, c)      # boundary tiles may stick out of the box
            f_vec = force_field[a, b, c]
            force_x[l, t] += m * f_vec[1]
            force_y[l, t] += m * f_vec[2]
            force_z[l, t] += m * f_vec[3]
        end
    end
end

function apply_external_forces!(force::VectorFieldForce, grid::SparseGrid{T, S, M, BoxDomain}, dt::Real) where {T, S, M}
    domain = grid.domain
    force_field = force.force_field
    @assert size(force_field) == Tuple(domain.hi .- domain.lo .+ 1) "Force field dimensions must match the BoxDomain."

    n = grid.n_active_write
    n == 0 && return nothing
    state = grid.state_write

    backend = KernelAbstractions.get_backend(state.mass)
    dev_force_field = _to_backend(backend, force_field)

    kernel = sparse_vector_field_force_kernel!(backend)
    kernel(
        state.force.x, state.force.y, state.force.z,
        state.mass, grid.map_write.tile_keys,
        dev_force_field, domain.lo;
        ndrange=(TILE_VOLUME, n)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

apply_external_forces!(::VectorFieldForce, ::SparseGrid{T, S, M, OpenDomain}, ::Real) where {T, S, M} =
    error("VectorFieldForce needs a bounded grid: use a DenseGrid or a SparseGrid with a BoxDomain.")