struct QuadraticSpline <: AbstractShapeFunction end

@fastmath @inline function shapefunction(::QuadraticSpline, natural_coords::SVector{3, T}) where {T}
    Nx = max(one(T) - abs(natural_coords[1]), zero(T))
    Ny = max(one(T) - abs(natural_coords[2]), zero(T))
    Nz = max(one(T) - abs(natural_coords[3]), zero(T))

    N = Nx * Ny * Nz
    
    return N
end




@inline function get_support_base(::QuadraticSpline, grid_coords::SVector{3, T}) where {T}
    i = floor(Int, grid_coords[1] - T(0.5))
    j = floor(Int, grid_coords[2] - T(0.5))
    k = floor(Int, grid_coords[3] - T(0.5))

    return i, j, k
end

@inline get_support_offsets(::QuadraticSpline) = (0:2, 0:2, 0:2)




@inline function B_update(::QuadraticSpline, N, r_rel, v_I)
    return N * r_rel' * v_I
end

@inline function M_inv(::QuadraticSpline, inv_dx)
    return 4 * inv_dx^2
end
