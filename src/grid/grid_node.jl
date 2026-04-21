struct GridNode{T}
    m::T                    # Mass
    c::T                    # Soundspeed
    v::SVector{3, T}        # Velocity  
end


function Base.zero(::Type{GridNode{T}}) where {T}
    return GridNode(zero(T), zero(SVector{3, T}))
end