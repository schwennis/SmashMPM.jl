# ============================================================================ #
#                               EXTERNAL FORCES                                #
# ============================================================================ #
# External forces act on the write state of the grid after G2P2G:
#   p_I += m_I * f(x_I) * dt
#
# Every force is implemented for DenseGrid and SparseGrid:
#   - DenseGrid:  nodal arrays of size (Nx, Ny, Nz), node (i, j, k)
#   - SparseGrid: tile pool of size (TILE_VOLUME, max_tiles), only the
#                 columns 1:n_active_write are in use. The global node index
#                 of entry (l, t) is node_index(map_write.tile_keys[t], l).
#
# Node positions are always x_I = origin + dx * ((i, j, k) .- 1), for the
# sparse grid with possibly negative global indices (open domains).
#
# Note: VectorFieldForce is only defined for a SparseGrid with a BoxDomain 
# since an unbounded domain cannot be saved as an array.
# ============================================================================ #

abstract type AbstractExternalForce end

"""Position of the global node (i, j, k)."""
@inline function node_position(origin::SVector{3, T}, inv_dx::T, i, j, k) where {T}
    dx = one(T) / inv_dx
    return SVector{3, T}(origin[1] + dx * (i - 1),
                         origin[2] + dx * (j - 1),
                         origin[3] + dx * (k - 1))
end

"""Columns of the sparse tile pool that currently hold write data."""
@inline active_write(grid::SparseGrid, field) = view(field, :, 1:grid.n_active_write)


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

    # Skalare vorab berechnen, um Multiplikationen im GPU-Loop zu sparen
    gx_dt = T(force.g[1] * dt)
    gy_dt = T(force.g[2] * dt)
    gz_dt = T(force.g[3] * dt)

    state.momentum.x .+= state.mass .* gx_dt
    state.momentum.y .+= state.mass .* gy_dt
    state.momentum.z .+= state.mass .* gz_dt
end

function apply_external_forces!(force::ConstantGravity, grid::SparseGrid{T}, dt::Real) where {T}
    grid.n_active_write == 0 && return nothing
    state = grid.state_write
    mass  = active_write(grid, state.mass)

    gx_dt = T(force.g[1] * dt)
    gy_dt = T(force.g[2] * dt)
    gz_dt = T(force.g[3] * dt)

    active_write(grid, state.momentum.x) .+= mass .* gx_dt
    active_write(grid, state.momentum.y) .+= mass .* gy_dt
    active_write(grid, state.momentum.z) .+= mass .* gz_dt
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
@kernel function radial_inv_square_force_kernel!(momentum_x, momentum_y, momentum_z, mass, origin, inv_dx, center, F_0, dt)
    # Direkte 3D-Indexierung über GPU-Register (vermeidet div/mod)
    i, j, k = @index(Global, NTuple)

    m = mass[i, j, k]
    T = eltype(mass)

    # Nur auswerten, wenn am Knoten tatsächlich Masse anliegt
    if m > eps(T)
        pos = node_position(origin, inv_dx, i, j, k)
        force_vec = radial_inv_square_force(pos, center, F_0)

        momentum_x[i, j, k] += m * force_vec[1] * dt
        momentum_y[i, j, k] += m * force_vec[2] * dt
        momentum_z[i, j, k] += m * force_vec[3] * dt
    end
end

function apply_external_forces!(force::RadialInvSquareForceField, grid::DenseGrid{T, S}, dt::Real) where {T, S}
    state = grid.state_write

    backend = KernelAbstractions.get_backend(state.mass)
    kernel = radial_inv_square_force_kernel!(backend)

    kernel(
        state.momentum.x,
        state.momentum.y,
        state.momentum.z,
        state.mass,
        grid.origin,
        grid.inv_dx,
        force.center,
        force.F_0,
        T(dt);
        ndrange=size(state.mass) # 3D-Range
    )

    KernelAbstractions.synchronize(backend)
end

# --- SparseGrid --------------------------------------------------------------- #
@kernel function sparse_radial_inv_square_force_kernel!(momentum_x, momentum_y, momentum_z, mass, tile_keys,
                                                        origin, inv_dx, center, F_0, dt)
    l, t = @index(Global, NTuple)

    m = mass[l, t]
    T = eltype(mass)

    if m > eps(T)
        i, j, k = node_index(tile_keys[t], l)
        pos = node_position(origin, inv_dx, i, j, k)
        force_vec = radial_inv_square_force(pos, center, F_0)

        momentum_x[l, t] += m * force_vec[1] * dt
        momentum_y[l, t] += m * force_vec[2] * dt
        momentum_z[l, t] += m * force_vec[3] * dt
    end
end

function apply_external_forces!(force::RadialInvSquareForceField, grid::SparseGrid{T}, dt::Real) where {T}
    n = grid.n_active_write
    n == 0 && return nothing
    state = grid.state_write

    backend = KernelAbstractions.get_backend(state.mass)
    kernel = sparse_radial_inv_square_force_kernel!(backend)

    kernel(
        state.momentum.x, state.momentum.y, state.momentum.z,
        state.mass, grid.map_write.tile_keys,
        grid.origin, grid.inv_dx, force.center, force.F_0, T(dt);
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
@kernel function vector_field_force_kernel!(momentum_x, momentum_y, momentum_z, mass, force_field, dt)
    i, j, k = @index(Global, NTuple)

    m = mass[i, j, k]
    if m > eps(eltype(mass))
        f_vec = force_field[i, j, k]
        momentum_x[i, j, k] += m * f_vec[1] * dt
        momentum_y[i, j, k] += m * f_vec[2] * dt
        momentum_z[i, j, k] += m * f_vec[3] * dt
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
        state.momentum.x,
        state.momentum.y,
        state.momentum.z,
        state.mass,
        dev_force_field,
        T(dt);
        ndrange=size(state.mass)
    )

    KernelAbstractions.synchronize(backend)
end

# --- SparseGrid --------------------------------------------------------------- #
@kernel function sparse_vector_field_force_kernel!(momentum_x, momentum_y, momentum_z, mass, tile_keys,
                                                   force_field, lo, dt)
    l, t = @index(Global, NTuple)

    m = mass[l, t]
    if m > eps(eltype(mass))
        i, j, k = node_index(tile_keys[t], l)
        a, b, c = i - lo[1] + 1, j - lo[2] + 1, k - lo[3] + 1
        if checkbounds(Bool, force_field, a, b, c)      # boundary tiles may stick out of the box
            f_vec = force_field[a, b, c]
            momentum_x[l, t] += m * f_vec[1] * dt
            momentum_y[l, t] += m * f_vec[2] * dt
            momentum_z[l, t] += m * f_vec[3] * dt
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
        state.momentum.x, state.momentum.y, state.momentum.z,
        state.mass, grid.map_write.tile_keys,
        dev_force_field, domain.lo, T(dt);
        ndrange=(TILE_VOLUME, n)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

apply_external_forces!(::VectorFieldForce, ::SparseGrid{T, S, M, OpenDomain}, ::Real) where {T, S, M} =
    error("VectorFieldForce needs a bounded grid: use a DenseGrid or a SparseGrid with a BoxDomain.")