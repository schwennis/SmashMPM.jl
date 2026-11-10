abstract type AbstractGrid end


@kwdef struct GridNode{T}
    mass::T
    momentum::SVector{3, T}
    force::SVector{3, T}
    wave_speed::T  # maximum wave speed v_p + c_p around this node (for CFL condition)
end

function Base.zero(::Type{GridNode{T}}) where {T}
    return GridNode(mass=zero(T), momentum=zero(SVector{3, T}), force=zero(SVector{3, T}), wave_speed=zero(T))
end


# ---------------------------------------------------------------------------- #
#                         Interface for the grid itself                        #
# ---------------------------------------------------------------------------- #
"""
    max_wavespeed(grid)
Returns the maximum wave speed in the grid. Used for CFL condition.
"""
function max_wavespeed(grid::AbstractGrid)
    _not_implemented(:max_wavespeed, grid)    
end

"""
    grid_reset!(grid, particle_sets, spline)
Resets the grid for the next time step, as `grid_reset!(grid)`, and prepares the
write state for the next g2p2g step. For a SparseGrid this activates the tiles
around the current particle positions; for a DenseGrid it does nothing extra.
"""
function grid_reset!(grid::AbstractGrid, particle_sets, spline)
    return grid_reset!(grid)
end

"""
    grid_reset!(grid)
Resets the grid for the next time step. Swaps the read and write states and
resets the write state to zero.
"""
function grid_reset!(grid::AbstractGrid)
    _not_implemented(:grid_reset!, grid)
end


# ---------------------------------------------------------------------------- #
#                                  Dense Grid                                  #
# ---------------------------------------------------------------------------- #
include("grids/dense_grid.jl")


# ---------------------------------------------------------------------------- #
#                                  Sparse Grid                                 #
# ---------------------------------------------------------------------------- #
include("grids/tile_keys.jl")
include("grids/tile_map.jl")
include("grids/sparse_grid.jl")