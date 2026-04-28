struct NoSlipBoundary{T} <: AbstractBoundaryCondition
    mask::Array{Bool, 3}    # Mask indicating which grid nodes will be set to zero velocity
end

function apply_boundary_condition!(grid::DenseGrid{T, S}, bc::NoSlipBoundary{T}) where {T, S}
    state = grid.state_new

    # Apply no-slip condition by setting velocity to zero at masked nodes
    for i in eachindex(state.v)
        if bc.mask[i]
            state.v[i] .= 0.0
        end
    end
end