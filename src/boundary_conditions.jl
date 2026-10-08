abstract type AbstractBoundaryCondition end

# ---------------------------------------------------------------------------- #
#                             No Boundary Condition                            #
# ---------------------------------------------------------------------------- #
struct NullBoundaryCondition <: AbstractBoundaryCondition end

function apply_boundary_condition!(::NullBoundaryCondition, grid::DenseGrid{T, S}) where {T, S}
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
