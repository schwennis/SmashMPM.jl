
@inline function get_grid_position(pos_p::SVector{3, T}, inv_dx::T, origin::SVector{3, T}, padding::Int) where {T}
    return (pos_p - origin) * inv_dx + (padding + one(T))  # +1 for 1-based indexing
end
