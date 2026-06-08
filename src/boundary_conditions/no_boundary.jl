function apply_boundary_condition!(::NoBoundaryCondition, grid::DenseGrid{T, S}) where {T, S}
    # No boundary condition to apply
    return
end