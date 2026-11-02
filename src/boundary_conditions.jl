abstract type AbstractBoundaryCondition end

# ---------------------------------------------------------------------------- #
#                             No Boundary Condition                            #
# ---------------------------------------------------------------------------- #
struct NullBoundaryCondition <: AbstractBoundaryCondition end

function apply_boundary_condition!(::NullBoundaryCondition, grid::AbstractGrid)
    # No boundary condition to apply
    return nothing
end

# ---------------------------------------------------------------------------- #
#                          No Slip Boundary Condition                          #
# ---------------------------------------------------------------------------- #
# No Mask - Sets padding momentum to zero
struct NoSlipBoundary <: AbstractBoundaryCondition end

@kernel function noslip_boundary_kernel!(momentum_x, momentum_y, momentum_z, padding, nx, ny, nz)
    i, j, k = @index(Global, NTuple)
    T = eltype(momentum_x)

    # Analytischer Randcheck: Liegt der Knoten im Padding-Bereich?
    is_boundary = (i <= padding || i > nx - padding ||
                   j <= padding || j > ny - padding ||
                   k <= padding || k > nz - padding)

    if is_boundary
        momentum_x[i, j, k] = zero(T)
        momentum_y[i, j, k] = zero(T)
        momentum_z[i, j, k] = zero(T)
    end
end

function apply_boundary_condition!(::NoSlipBoundary, grid::DenseGrid{T, S}) where {T, S}
    state = grid.state_write
    dims = size(state.mass)
    padding = grid.padding

    backend = KernelAbstractions.get_backend(state.mass)
    kernel = noslip_boundary_kernel!(backend)

    kernel(
        state.momentum.x,
        state.momentum.y,
        state.momentum.z,
        padding,
        dims[1], dims[2], dims[3];
        ndrange=dims
    )

    KernelAbstractions.synchronize(backend)
end


# ---------------------------------------------------------------------------- #
#                      No Slip Boundary on the Sparse Grid                     #
# ---------------------------------------------------------------------------- #
# Only meaningful on a BoxDomain: nodes within `padding` of the box faces
# (and nodes of boundary tiles sticking out of the box) are set to zero momentum.
@kernel function sparse_noslip_boundary_kernel!(momentum_x, momentum_y, momentum_z, tile_keys, lo, hi, padding)
    l, t = @index(Global, NTuple)
    T = eltype(momentum_x)

    i, j, k = node_index(tile_keys[t], l)
    is_boundary = (i < lo[1] + padding) | (i > hi[1] - padding) |
                  (j < lo[2] + padding) | (j > hi[2] - padding) |
                  (k < lo[3] + padding) | (k > hi[3] - padding)

    if is_boundary
        momentum_x[l, t] = zero(T)
        momentum_y[l, t] = zero(T)
        momentum_z[l, t] = zero(T)
    end
end

function apply_boundary_condition!(::NoSlipBoundary, grid::SparseGrid{T, S, M, BoxDomain}) where {T, S, M}
    n = grid.n_active_write
    n == 0 && return nothing
    state  = grid.state_write
    domain = grid.domain

    backend = KernelAbstractions.get_backend(state.mass)
    kernel  = sparse_noslip_boundary_kernel!(backend)
    kernel(
        state.momentum.x, state.momentum.y, state.momentum.z,
        grid.map_write.tile_keys, domain.lo, domain.hi, grid.padding;
        ndrange=(TILE_VOLUME, n)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

apply_boundary_condition!(::NoSlipBoundary, ::SparseGrid{T, S, M, OpenDomain}) where {T, S, M} =
    error("NoSlipBoundary needs walls: use a SparseGrid with a BoxDomain, " *
          "or NullBoundaryCondition for open boundaries.")