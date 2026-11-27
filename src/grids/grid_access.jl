# ---------------------------------------------------------------------------- #
#                                  Grid Access                                 #
# ---------------------------------------------------------------------------- #
# Device-side interface between the particle kernels and the grid types.
# Kernels never index a grid state directly; they go through the functions in
# this file, so one kernel serves every grid type. Julia compiles the kernels
# separately for every access type, so all dense / sparse decisions are made at
# compile time (zero cost: no runtime dispatch, no extra branches or loads).
#
# Used by: g2p2g! (solver/g2p2g.jl), initial_p2g! (setup/initial_p2g.jl),
#          extract_velocities (export/extract_velocities.jl)
#
# Included from grid.jl after dense_grid.jl and sparse_grid.jl.
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                     Stencil tiles (SparseGrid lookup cache)                  #
# ---------------------------------------------------------------------------- #
"""
Presaved tile indices of the (up to 8) tiles that contain the nodes of one stencil.
`t[1 + bx + 2by + 4bz]` is the tile index (0 = tile not active) of the tile
that is `bx, by, bz ∈ {0, 1}` tiles away from the tile of the lowest stencil node.
"""
struct StencilTiles
    t::NTuple{8, Int32}
    ti::Int     # tile coordinates of the lowest stencil node
    tj::Int
    tk::Int
end

@inline function find_stencil_tiles(map::AbstractTileMap, base, lo_off::Int, hi_off::Int)
    ti0, tj0, tk0 = tile_coord(base[1] + lo_off), tile_coord(base[2] + lo_off), tile_coord(base[3] + lo_off)
    ti1, tj1, tk1 = tile_coord(base[1] + hi_off), tile_coord(base[2] + hi_off), tile_coord(base[3] + hi_off)

    tiles = ntuple(Val(8)) do n
        bx = (n - 1) & 1
        by = ((n - 1) >> 1) & 1
        bz = (n - 1) >> 2

        # the upper tile of an axis is only needed if it differs from the lower one
        needed = ((bx == 0) | (ti1 != ti0)) & ((by == 0) | (tj1 != tj0)) & ((bz == 0) | (tk1 != tk0))
        ti = bx == 0 ? ti0 : ti1
        tj = by == 0 ? tj0 : tj1
        tk = bz == 0 ? tk0 : tk1

        if needed && tile_in_key_range(ti, tj, tk)
            Int32(find_tile(map, tile_key(ti, tj, tk)))
        else
            Int32(0)
        end
    end
    return StencilTiles(tiles, ti0, tj0, tk0)
end

"""Tile index of node (i, j, k) from the cached stencil tiles (0 = not active)."""
@inline function node_tile(st::StencilTiles, i, j, k)
    bx = Int(tile_coord(i) != st.ti)
    by = Int(tile_coord(j) != st.tj)
    bz = Int(tile_coord(k) != st.tk)
    return Int(st.t[1 + bx + 2 * by + 4 * bz])
end


# ---------------------------------------------------------------------------- #
#                                Access interface                              #
# ---------------------------------------------------------------------------- #
# A grid access is a lightweight view of ONE grid state (read or write) that is
# passed into the kernel. A new grid type only has to provide
#
#   read_access(grid), write_access(grid)             host: build the views
#   position_ok(access, grid_pos)    -> Bool          can the particle be mapped?
#   stencil_cache(access, spline, base)               per-stencil lookup data
#   node_slot(access, cache, i, j, k) -> (idx, status)
#   record_miss!(access)                              count a lost write / particle
#   check_stencil(grid, spline)                       host: sanity checks
#
# `idx` is a tuple that indexes the state arrays directly (state.mass[idx...]).

struct DenseAccess{S}
    state::S                # StructArray{GridNode}, indexed by (i, j, k)
end

struct SparseAccess{S, M <: AbstractTileMap, D <: AbstractDomain}
    state::S                # StructArray{GridNode}, indexed by (l, t)
    map::M                  # tile map that belongs to `state`
    domain::D
end

Adapt.@adapt_structure DenseAccess
Adapt.@adapt_structure SparseAccess

read_access(g::DenseGrid)   = DenseAccess(g.state_read)
write_access(g::DenseGrid)  = DenseAccess(g.state_write)
read_access(g::SparseGrid)  = SparseAccess(g.state_read,  g.map_read,  g.domain)
write_access(g::SparseGrid) = SparseAccess(g.state_write, g.map_write, g.domain)

# node status returned by node_slot
const NODE_OK       = 0     # node exists
const NODE_CLIPPED  = 1     # outside of the grid / box: silently skipped
const NODE_INACTIVE = 2     # inside the domain, but its tile is not active

# ---- position check --------------------------------------------------------- #
# DenseGrid: no check (as before, out-of-grid nodes are clipped in node_slot)
@inline position_ok(::DenseAccess, grid_pos)  = true
@inline position_ok(::SparseAccess, grid_pos) = grid_pos_representable(grid_pos)

# ---- miss counter ----------------------------------------------------------- #
@inline record_miss!(::DenseAccess)   = nothing
@inline record_miss!(g::SparseAccess) = record_miss!(g.map)

# ---- per-stencil cache ------------------------------------------------------ #
# SparseGrid: the (up to 8) tile lookups of a stencil are done once, not per node
@inline stencil_cache(::DenseAccess, spline, base) = nothing
@inline function stencil_cache(g::SparseAccess, spline, base)
    offsets = get_support_offsets(spline)[1]
    return find_stencil_tiles(g.map, base, first(offsets), last(offsets))
end

# ---- node lookup ------------------------------------------------------------ #
@inline function node_slot(g::DenseAccess, ::Nothing, i, j, k)
    inside = checkbounds(Bool, g.state.mass, i, j, k)
    return (i, j, k), ifelse(inside, NODE_OK, NODE_CLIPPED)
end

@inline function node_slot(g::SparseAccess, st::StencilTiles, i, j, k)
    l = local_index(i, j, k)
    node_in_domain(g.domain, i, j, k) || return (l, 0), NODE_CLIPPED   # OpenDomain: always true
    t = node_tile(st, i, j, k)
    return (l, t), ifelse(t == 0, NODE_INACTIVE, NODE_OK)
end

# ---- node data -------------------------------------------------------------- #
@inline node_mass(g, idx)     = g.state.mass[idx...]
@inline node_momentum(g, idx) = g.state.momentum[idx...]
@inline node_force(g, idx)    = g.state.force[idx...]
@inline node_e_mix(g, idx)    = g.state.e_mix[idx...]

"""
    reference_velocity(g, cache, i, j, k, T) -> SVector{3}

Velocity of node (i, j, k) in the grid state `g` (0 if the node does not exist
or has no mass). Used as the per-node reference velocity of the heat bookkeeping:
the P2G (all particles of a node) and `mixing_loss!` see the same value.
"""
@inline function reference_velocity(g, cache, i, j, k, ::Type{T}) where {T}
    idx, status = node_slot(g, cache, i, j, k)
    status == NODE_OK || return zero(SVector{3, T})
    m = node_mass(g, idx)
    return m > sqrt(floatmin(T)) ? node_momentum(g, idx) / m : zero(SVector{3, T})
end

# explicit index forms, so the atomic macro never sees a splat
@inline atomic_add!(a, (i, j, k)::NTuple{3, Int}, v) = (@atomic :monotonic a[i, j, k] += v; nothing)
@inline atomic_add!(a, (l, t)::NTuple{2, Int}, v)    = (@atomic :monotonic a[l, t] += v; nothing)

"""Atomically add mass `dm` and momentum `p` to a node (initial P2G, no stress yet)."""
@inline function scatter!(g, idx, dm, p::SVector{3})
    s = g.state
    atomic_add!(s.mass,       idx, dm)
    atomic_add!(s.momentum.x, idx, p[1])
    atomic_add!(s.momentum.y, idx, p[2])
    atomic_add!(s.momentum.z, idx, p[3])
    return nothing
end

"""Atomically add mass `dm`, momentum `p` and force `f` to a node (P2G of the G2P2G)."""
@inline function scatter!(g, idx, dm, p::SVector{3}, f::SVector{3})
    scatter!(g, idx, dm, p)
    s = g.state
    atomic_add!(s.force.x, idx, f[1])
    atomic_add!(s.force.y, idx, f[2])
    atomic_add!(s.force.z, idx, f[3])
    return nothing
end

"""Atomically add the kinetic energy `e` that a particle puts onto a node (heat bookkeeping)."""
@inline function scatter_energy!(g, idx, e)
    atomic_add!(g.state.e_mix, idx, e)
    return nothing
end


# ---- host-side checks ------------------------------------------------------- #
check_stencil(::AbstractGrid, spline) = nothing
function check_stencil(::SparseGrid, spline)
    # The tile lookup assumes that a stencil spans at most 2 tiles per axis
    @assert length(get_support_offsets(spline)[1]) <= TILE_SIZE "Shape function support is wider than TILE_SIZE"
    return nothing
end
