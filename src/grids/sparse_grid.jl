# ---------------------------------------------------------------------------- #
#                                  Sparse Grid                                 #
# ---------------------------------------------------------------------------- #
# Tile-based sparse grid after Qiu et al. 2023
#
# TILE_SIZE^3 nodes are organized into a tile (see tile_keys.jl).
# Only tiles interacting with particles are activated.
#
# Nodal data is stored in a 2D StructArray of size (TILE_VOLUME, max_tiles),
# where each column corresponds to a tile and each row to a node in the tile.
# Inside the tile, nodes are flattened linearly:
#   state.mass[l, t] = mass of node l in tile t
#   where l ∈ [1, TILE_VOLUME] and t ∈ [1, n_active]
#
# The tile index t is found from the global node index (i, j, k) via a tile map
# (AbstractTileMap, e.g. TileHashMap):  find_node(map, domain, i, j, k) -> (l, t)
#
# ---------------------------------------------------------------------------- #
# Domain / open boundaries
# ---------------------------------------------------------------------------- #
# `origin` is the position of node (1, 1, 1) and only defines the coordinate
# frame. Node indices may be <= 0 or arbitrarily large, the tile keys are
# signed and domain independent. Whether the grid is bounded is decided by
#
#   OpenDomain()          no bounds at all, particles may go anywhere
#                         (limited only by the key range, ±2^20 tiles per axis).
#                         Use with NullBoundaryCondition.
#   BoxDomain(lo, hi)     nodes lo..hi (inclusive) exist, everything outside is
#                         clipped as in the DenseGrid. Needed for wall boundary
#                         conditions (NoSlipBoundary uses the padding at the box faces).
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    Domains                                   #
# ---------------------------------------------------------------------------- #
abstract type AbstractDomain end

"""Unbounded domain (open boundary conditions)."""
struct OpenDomain <: AbstractDomain end

"""Bounded domain of the node indices lo..hi (inclusive, 1-based global indices)."""
struct BoxDomain <: AbstractDomain
    lo::SVector{3, Int}
    hi::SVector{3, Int}
end
"""Box of the nodes 1..N, i.e. the same nodes as a DenseGrid of size N."""
BoxDomain(N::SVector{3, Int}) = BoxDomain(SVector(1, 1, 1), N)

@inline node_in_domain(::OpenDomain, i, j, k) = true
@inline node_in_domain(d::BoxDomain, i, j, k) =
    (d.lo[1] <= i <= d.hi[1]) & (d.lo[2] <= j <= d.hi[2]) & (d.lo[3] <= k <= d.hi[3])

"""Clip a node range [lo, hi] to the domain."""
@inline clip_range(::OpenDomain, lo, hi) = (lo, hi)
@inline clip_range(d::BoxDomain, lo, hi) = (max.(lo, d.lo), min.(hi, d.hi))

"""Upper bound for the number of tiles the domain can ever activate."""
max_domain_tiles(::OpenDomain) = Int(typemax(Int32))    # tile indices are Int32
max_domain_tiles(d::BoxDomain) = prod(tile_coord.(d.hi) .- tile_coord.(d.lo) .+ 1)

function check_domain(d::BoxDomain)
    @assert all(d.lo .<= d.hi) "BoxDomain: lo must be <= hi"
    tlo, thi = tile_coord.(d.lo), tile_coord.(d.hi)
    @assert tile_in_key_range(tlo...) && tile_in_key_range(thi...) "BoxDomain too large for the tile key range"
    return nothing
end
check_domain(::OpenDomain) = nothing


# ---------------------------------------------------------------------------- #
#                                  SparseGrid                                  #
# ---------------------------------------------------------------------------- #
mutable struct SparseGrid{T, S<:StructArray, M<:AbstractTileMap, D<:AbstractDomain} <: AbstractGrid
    state_read::S               # tile pool, g2p2g reads from here
    state_write::S              # tile pool, g2p2g writes to here
    map_read::M                 # tile key -> tile index for state_read
    map_write::M                # tile key -> tile index for state_write
    n_active_read::Int          # tiles 1:n_active_read of state_read are in use
    n_active_write::Int         # tiles 1:n_active_write of state_write are in use

    domain::D                   # OpenDomain() or BoxDomain(lo, hi)
    padding::Int                # padding width in nodes (only used by wall BCs on a BoxDomain)

    origin::SVector{3, T}       # position of node (1, 1, 1), defines the coordinate frame only
    inv_dx::T                   # 1/dx, where dx is the node spacing
end


function _allocate_tile_pool(backend, ::Type{T}, max_tiles::Int) where {T}
    new_field() = _to_backend(backend, zeros(T, TILE_VOLUME, max_tiles))
    momentum = StructArray{SVector{3, T}}((x=new_field(), y=new_field(), z=new_field()))
    return StructArray{GridNode{T}}((mass=new_field(), momentum=momentum, wave_speed=new_field()))
end


"""
    SparseGrid(dx, origin; domain=OpenDomain(), padding=0,
               tile_map=TileHashMap, backend=CPU(), max_tiles=4096)

Sparse tile grid. With the default `OpenDomain()` the grid has no bounds
(open boundary conditions). `max_tiles` is only the initial pool size; the
pool grows automatically in `activate_tiles!`.
"""
function SparseGrid(dx::T, origin::SVector{3, T};
                    domain::AbstractDomain=OpenDomain(),
                    padding::Int=0,
                    tile_map::Type{<:AbstractTileMap}=TileHashMap,
                    backend=CPU(),
                    max_tiles::Int=4096) where {T}
    check_domain(domain)
    max_tiles = clamp(max_tiles, 1, max_domain_tiles(domain))

    return SparseGrid(
        _allocate_tile_pool(backend, T, max_tiles),
        _allocate_tile_pool(backend, T, max_tiles),
        new_tile_map(tile_map, backend, max_tiles),
        new_tile_map(tile_map, backend, max_tiles),
        0, 0,
        domain, padding,
        origin, one(T) / dx,
    )
end

"""
    SparseGrid(dx, N, origin, padding=2, tile_map=TileHashMap, backend=CPU(); max_tiles=4096)

Bounded sparse grid with the same nodes as `DenseGrid(dx, N, origin, padding, backend)`.
"""
function SparseGrid(dx::T, N::SVector{3, Int}, origin::SVector{3, T}, padding::Int=2,
                    tile_map::Type{<:AbstractTileMap}=TileHashMap, backend=CPU();
                    max_tiles::Int=4096) where {T}
    return SparseGrid(dx, origin; domain=BoxDomain(N), padding, tile_map, backend, max_tiles)
end


# ---------------------------------------------------------------------------- #
#                             Helpers for indexing                             #
# ---------------------------------------------------------------------------- #
# Largest |grid coordinate| whose stencil (+ margin) still fits into the key range.
const MAX_GRID_COORD = TILE_SIZE * KEY_BIAS - 4 * TILE_SIZE

"""
False for positions outside the key range and for NaN / Inf positions.
Such particles cannot be mapped to tiles and are counted as missed.
"""
@inline grid_pos_representable(grid_pos::SVector{3, T}) where {T} =
    (abs(grid_pos[1]) < T(MAX_GRID_COORD)) &
    (abs(grid_pos[2]) < T(MAX_GRID_COORD)) &
    (abs(grid_pos[3]) < T(MAX_GRID_COORD))

"""
    find_node(map, domain, i, j, k) -> (l, t)

Converts global node indices (1-based, may be <= 0) to the local node index
l ∈ 1:TILE_VOLUME and the tile index t. Returns t = 0 if the tile is not
active or the node lies outside the domain.
"""
@inline function find_node(map::AbstractTileMap, domain::AbstractDomain, i, j, k)
    l = local_index(i, j, k)
    node_in_domain(domain, i, j, k) || return l, 0
    ti, tj, tk = tile_coord(i), tile_coord(j), tile_coord(k)
    tile_in_key_range(ti, tj, tk)   || return l, 0
    return l, find_tile(map, tile_key(ti, tj, tk))
end

"""Unclipped node range [lo, hi] of the spline support around `grid_pos`, widened by `margin`."""
@inline function stencil_node_range(spline, grid_pos, margin)
    base    = SVector(get_support_base(spline, grid_pos))
    offsets = get_support_offsets(spline)
    lo = base .+ SVector(first.(offsets)) .- margin
    hi = base .+ SVector(last.(offsets))  .+ margin
    return lo, hi
end


# ---------------------------------------------------------------------------- #
#                         Tile activation (sparse map)                         #
# ---------------------------------------------------------------------------- #
@kernel function activate_tiles_kernel!(map, positions, origin, inv_dx, domain, spline, margin)
    p_idx = @index(Global, Linear)

    grid_pos = grid_position(positions[p_idx], inv_dx, origin)

    if grid_pos_representable(grid_pos)
        lo, hi = stencil_node_range(spline, grid_pos, margin)
        lo, hi = clip_range(domain, lo, hi)

        if all(lo .<= hi)       # stencil (partly) inside the domain
            for tk in tile_coord(lo[3]):tile_coord(hi[3]),
                tj in tile_coord(lo[2]):tile_coord(hi[2]),
                ti in tile_coord(lo[1]):tile_coord(hi[1])

                insert_tile!(map, tile_key(ti, tj, tk))
            end
        end
    else
        record_miss!(map)       # particle escaped the key range or is NaN
    end
end

"""
    register_tiles!(map, grid, positions, spline, margin)

Register the tiles around the given positions in the sparse grid.
"""
function register_tiles!(map::AbstractTileMap, grid::SparseGrid, positions, spline, margin::Int=ACTIVATION_MARGIN)
    isempty(positions) && return nothing
    backend = KernelAbstractions.get_backend(map.tile_keys)
    kernel = activate_tiles_kernel!(backend)
    kernel(map, positions, grid.origin, grid.inv_dx, grid.domain, spline, margin;
           ndrange=length(positions))
    KernelAbstractions.synchronize(backend)
    return nothing
end


"""Headroom of a grown tile pool over the counted tiles (as in `build_mpm_model`)."""
const TILE_POOL_HEADROOM = 5 // 4

pool_tiles(grid::SparseGrid) = size(grid.state_write.mass, 2)

"""
    activate_tiles!(grid, particle_sets, spline)

Allocate the write-grid tiles that the coming G2P2G step can scatter into.
Grows the (still empty) write pool if it is too small.

Growing happens in at most two stages:
  1. If the tile map was saturated, `tile_count` is only a lower bound. Then only
     the (small) map is enlarged and the tiles are counted again; the large
     nodal pool is not touched.
  2. With an exact count `n`, pool and map are reallocated for `n` tiles plus
     `TILE_POOL_HEADROOM`, and the tiles are registered once more.
"""
function activate_tiles!(grid::SparseGrid{T}, particle_sets, spline) where {T}
    backend  = KernelAbstractions.get_backend(grid.state_write.mass)
    MapType  = typeof(grid.map_write)
    domain_max = max_domain_tiles(grid.domain)

    while true
        foreach(particle_sets) do p_set
            register_tiles!(grid.map_write, grid, p_set.particles.pos, spline, ACTIVATION_MARGIN)
        end

        n = tile_count(grid.map_write)
        if n <= pool_tiles(grid) && max_tiles(grid.map_write) == pool_tiles(grid)
            grid.n_active_write = n
            return n
        end

        if n >= count_limit(grid.map_write)
            # Stage 1: saturated map, true tile count unknown. Recount with a larger map.
            new_map_max = min(4 * max_tiles(grid.map_write), domain_max)
            @debug "SparseGrid: tile map saturated at $n tiles, recounting with room for $new_map_max"
            grid.map_write = new_tile_map(MapType, backend, new_map_max)
            continue
        end

        # Stage 2: exact count. The write grid holds no data yet, so just replace it.
        new_max = n <= pool_tiles(grid) ? pool_tiles(grid) :     # fits after all: keep the pool
                  min(max(ceil(Int, n * TILE_POOL_HEADROOM), n), domain_max)
        if new_max != pool_tiles(grid)
            @info "SparseGrid: growing tile pool from $(pool_tiles(grid)) to $new_max tiles ($n active)"
            grid.state_write = _allocate_tile_pool(backend, T, new_max)
        end
        grid.map_write = new_tile_map(MapType, backend, new_max)
    end
end


"""
    count_tiles(positions, dx, origin, spline; domain=OpenDomain(), margin=ACTIVATION_MARGIN)

Count (on the host) the number of tiles that would be activated by the given
positions. Useful to choose `max_tiles` up front.
"""
function count_tiles(positions, dx::T, origin::SVector{3, T}, spline;
                     domain::AbstractDomain=OpenDomain(),
                     margin::Int=ACTIVATION_MARGIN) where {T}
    inv_dx = one(T) / dx
    keys   = Set{TileKey}()
    for pos in positions
        grid_pos = grid_position(pos, inv_dx, origin)
        grid_pos_representable(grid_pos) || continue
        lo, hi = clip_range(domain, stencil_node_range(spline, grid_pos, margin)...)
        all(lo .<= hi) || continue
        for tk in tile_coord(lo[3]):tile_coord(hi[3]),
            tj in tile_coord(lo[2]):tile_coord(hi[2]),
            ti in tile_coord(lo[1]):tile_coord(hi[1])

            push!(keys, tile_key(ti, tj, tk))
        end
    end
    return length(keys)
end

# Bounded variant with the DenseGrid signature
count_tiles(positions, dx::T, N::SVector{3, Int}, origin::SVector{3, T}, spline;
            margin::Int=ACTIVATION_MARGIN) where {T} =
    count_tiles(positions, dx, origin, spline; domain=BoxDomain(N), margin)


# ---------------------------------------------------------------------------- #
#                         Grid interface (like DenseGrid)                      #
# ---------------------------------------------------------------------------- #
function max_wavespeed(grid::SparseGrid{T}) where {T}
    grid.n_active_read == 0 && return zero(T)
    return maximum(view(grid.state_read.wave_speed, :, 1:grid.n_active_read))
end

"""
    grid_reset!(grid::SparseGrid, particle_sets, spline)

Swap read <-> write, clear the new write state and activate the tiles that
the next G2P2G step can write to (from the current particle positions).
This is the variant used by `step!`.

`grid_reset!(grid)` alone only swaps and clears; the write grid then has no
active tiles until `activate_tiles!` is called.
"""
function grid_reset!(grid::SparseGrid, particle_sets, spline)
    grid_reset!(grid)
    activate_tiles!(grid, particle_sets, spline)
    return nothing
end

function grid_reset!(grid::SparseGrid{T}) where {T}
    n_missed = missed_count(grid.map_write)
    n_missed > 0 && @warn "SparseGrid: $n_missed lost P2G writes / particles. Either a particle moved " *
                          "more than $ACTIVATION_MARGIN cell(s) in one step, or it left the " *
                          "representable key range (±$(KEY_BIAS * TILE_SIZE) nodes per axis)." maxlog=10

    # Swap read <-> write. Data, hash map and tile count always travel together.
    grid.state_read,    grid.state_write    = grid.state_write,    grid.state_read
    grid.map_read,      grid.map_write      = grid.map_write,      grid.map_read
    grid.n_active_read, grid.n_active_write = grid.n_active_write, grid.n_active_read

    # Clear the new write grid. Only the tiles that were in use contain data.
    used  = 1:grid.n_active_write
    state = grid.state_write
    for field in (state.mass, state.wave_speed, state.momentum.x, state.momentum.y, state.momentum.z)
        fill!(view(field, :, used), zero(T))
    end
    clear!(grid.map_write)
    grid.n_active_write = 0
    return nothing
end


# ---------------------------------------------------------------------------- #
#                      Conversion to DenseGrid (export, tests)                 #
# ---------------------------------------------------------------------------- #
"""
    to_dense(grid::SparseGrid) -> DenseGrid (on the CPU)

Copies both grid states into a DenseGrid. Inactive tiles become zero.

- `BoxDomain`:  the dense grid covers exactly the box, its origin is the
                position of node `domain.lo`.
- `OpenDomain`: the dense grid covers the bounding box of all active tiles
                (of both states), its origin is shifted accordingly.

Meant for export and debugging, not for every time step.
Refuses to allocate more than `max_nodes` nodes (per state), since with an
OpenDomain widely separated particles span an arbitrarily large box.
"""
function to_dense(grid::SparseGrid{T}; max_nodes::Int=200_000_000) where {T}
    keys_read  = Array(view(grid.map_read.tile_keys,  1:grid.n_active_read))
    keys_write = Array(view(grid.map_write.tile_keys, 1:grid.n_active_write))

    lo, hi = _dense_node_box(grid.domain, vcat(keys_read, keys_write))
    N      = hi .- lo .+ 1
    prod(N) <= max_nodes || error("to_dense: bounding box of $(Tuple(N)) = $(prod(N)) nodes exceeds " *
                                  "max_nodes = $max_nodes. Pass a larger max_nodes if this is intended.")
    dx     = one(T) / grid.inv_dx
    origin = grid.origin .+ T.(lo .- 1) .* dx     # position of global node lo

    state_read  = _to_dense_state(grid.state_read,  keys_read,  lo, N)
    state_write = _to_dense_state(grid.state_write, keys_write, lo, N)
    return DenseGrid(state_read, state_write, grid.padding, origin, grid.inv_dx)
end

_dense_node_box(d::BoxDomain, keys) = (d.lo, d.hi)

function _dense_node_box(::OpenDomain, keys)
    isempty(keys) && return SVector(1, 1, 1), SVector(1, 1, 1)
    tmin = SVector(tile_coords(first(keys)))
    tmax = tmin
    for key in keys
        t = SVector(tile_coords(key))
        tmin, tmax = min.(tmin, t), max.(tmax, t)
    end
    return 1 .+ TILE_SIZE .* tmin, TILE_SIZE .* (tmax .+ 1)
end

function _to_dense_state(state, tile_keys, lo::SVector{3, Int}, N::SVector{3, Int})
    T     = eltype(state.mass)
    dense = _allocate_grid_state(CPU(), T, N)
    mass, wave_speed = Array(state.mass), Array(state.wave_speed)
    px, py, pz = Array(state.momentum.x), Array(state.momentum.y), Array(state.momentum.z)

    for t in eachindex(tile_keys), l in 1:TILE_VOLUME
        i, j, k = node_index(tile_keys[t], l)
        a, b, c = i - lo[1] + 1, j - lo[2] + 1, k - lo[3] + 1      # dense indices
        ((1 <= a <= N[1]) & (1 <= b <= N[2]) & (1 <= c <= N[3])) || continue   # tile may stick out of the box

        dense.mass[a, b, c]       = mass[l, t]
        dense.wave_speed[a, b, c] = wave_speed[l, t]
        dense.momentum.x[a, b, c] = px[l, t]
        dense.momentum.y[a, b, c] = py[l, t]
        dense.momentum.z[a, b, c] = pz[l, t]
    end
    return dense
end