struct GridNode{T}
    m::T                    # Mass
    wave_speed::T           # max(v_p + c_p) for all particles contributing to this node
    p::SVector{3, T}        # Velocity  
end


function Base.zero(::Type{GridNode{T}}) where {T}
    return GridNode(zero(T), zero(T), zero(SVector{3, T}))
end