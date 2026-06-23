abstract type AbstractBoundaryCondition end

# ---------------------------------------------------------------------------- #
#                             No Boundary Condition                            #
# ---------------------------------------------------------------------------- #
struct NoBoundaryCondition <: AbstractBoundaryCondition end

function apply_boundary_condition!(::NoBoundaryCondition, grid::DenseGrid{T, S}) where {T, S}
    # No boundary condition to apply
    return
end

# ---------------------------------------------------------------------------- #
#                          No Slip Boundary Condition                          #
# ---------------------------------------------------------------------------- #
struct NoSlipBoundary{T} <: AbstractBoundaryCondition
    mask::Array{Bool, 3}    # Mask indicating which grid nodes will be set to zero velocity
end

function NoSlipBoundary(grid::DenseGrid{T,S}) where {T, S} 
    padding = grid.padding
    N = size(grid.state_new.mass) .- 2*padding  # Original grid size without padding
    mask = falses(size(grid.state_new.mass))  # Initialize mask with false
    # Set mask to true for  ghost nodes (padding region)
    mask[1:padding, :, :] .= true
    mask[end-padding+1:end, :, :] .= true
    mask[:, 1:padding, :] .= true
    mask[:, end-padding+1:end, :] .= true
    mask[:, :, 1:padding] .= true
    mask[:, :, end-padding+1:end] .= true

    return NoSlipBoundary{T}(mask)
end

function apply_boundary_condition!(grid::DenseGrid{T, S}, bc::NoSlipBoundary{T}) where {T, S}
    state = grid.state_new

    # Apply no-slip condition by setting velocity to zero at masked nodes
    state.momentum .= ifelse.(bc.mask, (zero(SVector{3, T}),), state.momentum)
end