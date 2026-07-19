abstract type AbstractGrid end


struct GridNode{T}
    mass::T
    momentum::SVector{3, T}
    wave_speed::T   # maximum wave speed v_p + c_p around this node (for CFL condition)
end

function Base.zero(::Type{GridNode{T}}) where {T}
    return GridNode(zero(T), zero(SVector{3, T}), zero(T))
end


# ---------------------------------------------------------------------------- #
#                                  Dense Grid                                  #
# ---------------------------------------------------------------------------- #
include("grids/dense_grid.jl")