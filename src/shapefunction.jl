function shapefunction(natural_coords::SVector{3, T}) where {T}
    Nx = max(one(T) - abs(natural_coords[1]), zero(T))
    Ny = max(one(T) - abs(natural_coords[2]), zero(T))
    Nz = max(one(T) - abs(natural_coords[3]), zero(T))

    N = Nx * Ny * Nz
    
    return N
end