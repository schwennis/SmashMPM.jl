abstract type AbstractGrid end


@kwdef struct GridNode{T}
    mass::T
    momentum::SVector{3, T}
    force::SVector{3, T}
    # Σ_p N ½ m |v_p + C_p r|²: kinetic energy the particles brought to the node in the
    # P2G. Only written by transfer schemes with heat = true (node P2G loss, g2p2g.jl).
    kin::T
end

function Base.zero(::Type{GridNode{T}}) where {T}
    return GridNode(mass=zero(T), momentum=zero(SVector{3, T}), force=zero(SVector{3, T}), kin=zero(T))
end


# ---------------------------------------------------------------------------- #
#                         Interface for the grid itself                        #
# ---------------------------------------------------------------------------- #
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


"""
Node of the MUSL buffer of the 2-pass G2P2G: mass and momentum re-mapped from the
kicked particles at the midpoint positions x^{n+1/2} (see solver/g2p2g.jl).
"""
struct MuslNode{T}
    mass::T
    momentum::SVector{3, T}
end

"""
Zero-filled `StructArray{MuslNode{T}}` with the array type and shape of `a`
(backend independent via `similar`).
"""
function _zero_musl_like(a::AbstractArray{T}) where {T}
    new_field() = fill!(similar(a), zero(T))
    momentum = StructArray{SVector{3, T}}((x=new_field(), y=new_field(), z=new_field()))
    return StructArray{MuslNode{T}}((mass=new_field(), momentum=momentum))
end

"""Zero the MUSL fields `musl` in the index range `cols...` (all of it by default)."""
function _clear_musl!(musl, cols...)
    T = eltype(musl.mass)
    for field in (musl.mass, musl.momentum.x, musl.momentum.y, musl.momentum.z)
        fill!(isempty(cols) ? field : view(field, :, cols...), zero(T))
    end
    return nothing
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


# ---------------------------------------------------------------------------- #
#                                  Grid Access                                 #
# ---------------------------------------------------------------------------- #
include("grids/grid_access.jl")
