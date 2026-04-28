struct NoBoundary<:AbstractBoundaryCondition end

function apply_boundary_condition!(grid::DenseGrid{T, S}, bc::NoBoundary) where {T, S}
    # No boundary condition to apply
    return
end