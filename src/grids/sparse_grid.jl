# ---------------------------------------------------------------------------- #
#                                  Sparse Grid                                 #
# ---------------------------------------------------------------------------- #
# Tile-based sparse grid after Qiu et al. 2023
#
# Here, the domain is divided into tiles:
# 
# TILE_SIZE^3 nodes are organized into a tile
# Tiles are organized into a grid
#
# TILE_SIZE is a compile time constant. 
const TILE_SIZE = 4                 # nodes per tile edge
const TILE_VOLUME = TILE_SIZE^3     # nodes per tile
# TILE_SIZE should always be a power of 2
# 
# Only tiles interacting with particles are activated.
# In each g2p2g step, the particles move at most 1 node (CFL-condition). 
# Since the P2G step uses the new position,
# the stencil of activated tiles must be extended by a margin in each direction:
const ACTIVATION_MARGIN = 1         # extra nodes around the stencil
# This Margin definition differs from the implementation in Qiu et al. 2023
# 
# Nodal data is stored in a 2D Structarray of size (TILE_VOLUME, max_nodes).
# Where each column corresponds to a tile, and each row corresponds to a node in the tile.
# Inside the tile, nodes are flattened linearly
# State access is done as follows:
#   state.mass[l, t] = mass of node l in tile t
#   where l ∈ [1, TILE_SIZE^3] and t ∈ [1, n_tiles]
#
# the tile index t is mapped from the node index i. 
# For this, a specific implementation of AbstractTileMap is used (eg a HashMap or an IndexMap)
# ---------------------------------------------------------------------------- #

mutable struct SparseGrid{T, S<:StructArray, M<:AbstractTileMap} <: AbstractGrid
    state_read::S               # tile pool, g2p2g reads from here
    state_write::S              # tile pool, g2p2g writes to here
    map_read::M                 # tile key -> tile index for state_read
    map_write::M                # tile key -> tile index for state_write
    n_active_read::Int          # tiles 1:n_active_read of state_read are in use
    n_active_write::Int         # tiles 1:n_active_write of state_write are in use

    n_nodes::SVector{3, Int}    # domain size in nodes (= size of an equivalent DenseGrid)
    n_tiles::SVector{3, Int}    # domain size in tiles (= n_nodes ./ TILE_SIZE)
    padding::Int                # padding width in nodes (used for boundary conditions)

    origin::SVector{3, T}       # position of node (1, 1, 1)
    inv_dx::T                   # 1/dx, where dx is the node spacing
end


function _allocate_tile_pool(backend, ::Type{T}, max_tiles::Int) where {T}
    new_field() = _to_backend(backend, zeros(T, TILE_VOLUME, max_tiles))
    momentum = StructArray{SVector{3, T}}((x=new_field(), y=new_field(), z=new_field()))
    return StructArray{GridNode{T}}((mass=new_field(), momentum=momentum, wave_speed=new_field()))
end


function SparseGrid(dx::T, N::SVector{3, Int}, origin::SVector{3, T}, padding::Int=0, TileMapType::M,
                    backend=CPU(); max_tiles::Int=4096) where {T, M<:AbstractTileMap}
    n_tiles = cld.(N, TILE_SIZE)
    @assert prod(n_tiles) < typemax(UInt32) "Domain too large for UInt32 tile keys"
    max_tiles = clamp(max_tiles, 1, prod(n_tiles))

    return SparseGrid(
        _allocate_tile_pool(backend, T, max_tiles),
        _allocate_tile_pool(backend, T, max_tiles),
        new_tile_map(M, backend, max_tiles),
        new_tile_map(M, backend, max_tiles),
        0, 0,
        N, n_tiles, padding,
        origin, one(T) / dx,
    )
end


# ---------------------------------------------------------------------------- #
#                             Helpers for indexing                             #
# ---------------------------------------------------------------------------- #
"""Checkbounds for node indices"""
@inline function node_in_bounds(n_nodes, i, j, k)
    return 1 <= i <= n_nodes[1] & 1 <= j <= n_nodes[2] & 1 <= k <= n_nodes[3]
end

"""Converts a node index to a tile coordinate (0-based) along one dimension"""
@inline tile_coord(i) = div((i - 1), TILE_SIZE)

"""Converts tile coordinates (zero-based) to a tile index (1-based)"""
@inline tile_key(ti, tj, tk, n_tiles) = (ti + n_tiles[1] * (tj + n_tiles[2] * tk)) % UInt32 # Lexiographic ordering

"""
    find_node(map, n_tiles, i, j, k) -> (l, t)

Converts global node indices to tile indices (1-based) and local node indices (1-based)
Returns t = 0 if the tile is not active.
"""
@inline function find_node(map::AbstractTileMap, n_tiles, i, j, k)
    ti, li = divrem(i - 1, TILE_SIZE)
    tj, lj = divrem(j - 1, TILE_SIZE)
    tk, lk = divrem(k - 1, TILE_SIZE)
    t = find_tile(map, tile_key(ti, tj, tk, n_tiles))
    l = 1 + li + TILE_SIZE * (lj + TILE_SIZE * lk)
    return l, t
end


"""
    node_index(key, l, n_tiles) -> (i, j, k)

Converts tile key and local node index to global node indices (1-based)
"""
@inline function node_index(key::UInt32, l, n_tiles)
    tk, rest = divrem(Int(key), n_tiles[1] * n_tiles[2])
    tj, ti   = divrem(rest, n_tiles[1])
    lk, rest = divrem(l - 1, TILE_SIZE * TILE_SIZE)
    lj, li   = divrem(rest, TILE_SIZE)
    return 1 + TILE_SIZE * ti + li, 1 + TILE_SIZE * tj + lj, 1 + TILE_SIZE * tk + lk
end

"""Returns the support offsets of a spline basis function considering a margin"""
@inline function support_offsets_with_margin(spline, grid_pos, margin, n_nodes)
    base    = SVector(get_support_base(spline, grid_pos))
    offsets = get_support_offsets(spline)
    lo = max.(base .+ SVector(first.(offsets)) .- margin, 1)
    hi = min.(base .+ SVector(last.(offsets))  .+ margin, n_nodes)
    return lo, hi
end

# ---------------------------------------------------------------------------- #
#                         Tile activation (sparse map)                         #
# ---------------------------------------------------------------------------- #
@kernel function activate_tiles_kernel!(map, positions, origin, inv_dx, n_nodes, n_tiles, spline, margin)
    p_idx = @index(Global, Linear)

    grid_pos = grid_position(positions[p_idx], inv_dx, origin)
    lo, hi = support_offsets_with_margin(spline, grid_pos, margin, n_nodes)

    if all(lo .<= hi)       # particle (partly) inside the domain
        for tk in tile_coord(lo[3]):tile_coord(hi[3]),
            tj in tile_coord(lo[2]):tile_coord(hi[2]),
            ti in tile_coord(lo[1]):tile_coord(hi[1])

            insert_tile!(map, tile_key(ti, tj, tk, n_tiles))
        end
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
    kernel(map, positions, grid.origin, grid.inv_dx, grid.n_nodes, grid.n_tiles, spline, margin;
           ndrange=length(positions))
    KernelAbstractions.synchronize(backend)
    return nothing
end


"""
    activate_tiles!(grid, particle_sets, spline)

Allocate the write-grid tiles that the coming G2P2G step can scatter into.
Grows the (still empty) write pool if it is too small.
"""
function activate_tiles!(grid::SparseGrid{T}, particle_sets, spline) where {T}
    while true
        foreach(particle_sets) do p_set
            register_tiles!(grid.map_write, grid, p_set.particles.pos, spline, ACTIVATION_MARGIN)
        end

        n = tile_count(grid.map_write)
        if n <= max_tiles(grid.map_write)
            grid.n_active_write = n
            return n
        end

        # Pool too small. The write grid holds no data yet, so just replace it.
        backend = KernelAbstractions.get_backend(grid.state_write.mass)
        new_max = min(2 * n, prod(grid.n_tiles))
        @info "SparseGrid: growing tile pool from $(max_tiles(grid.map_write)) to $new_max tiles"
        grid.state_write = _allocate_tile_pool(backend, T, new_max)
        grid.map_write   = new_tile_map(typeof(grid.map_write), backend, new_max, prod(grid.n_tiles))
    end
end


"""
    count_tiles(positions, inv_dx, N, origin, spline; margin=ACTIVATION_MARGIN)
Count the number of tiles that would be activated by the given positions.
"""
function count_tiles(positions, inv_dx::T, N::SVector{3, Int}, origin::SVector{3, T},
                     spline; margin::Int=ACTIVATION_MARGIN) where {T}
    n_tiles = cld.(N, TILE_SIZE)
    keys    = Set{UInt32}()
    for pos in positions
        lo, hi = support_offsets_with_margin(spline, grid_position(pos, inv_dx, origin), margin, N)
        all(lo .<= hi) || continue
        for tk in tile_coord(lo[3]):tile_coord(hi[3]),
            tj in tile_coord(lo[2]):tile_coord(hi[2]),
            ti in tile_coord(lo[1]):tile_coord(hi[1])

            push!(keys, tile_key(ti, tj, tk, n_tiles))
        end
    end
    return length(keys)
end


# ---------------------------------------------------------------------------- #
#                         Grid interface (like DenseGrid)                      #
# ---------------------------------------------------------------------------- #
function max_wavespeed(grid::SparseGrid{T}) where {T}
    grid.n_active_read == 0 && return zero(T)
    return maximum(view(grid.state_read.wave_speed, :, 1:grid.n_active_read))
end

function grid_reset!(grid::SparseGrid{T}) where {T}
    n_missed = missed_count(grid.map_write)
    n_missed > 0 && @warn "SparseGrid: $n_missed P2G writes hit an inactive tile. " *
                          "A particle moved more than $ACTIVATION_MARGIN cell(s) in one step." maxlog=10

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

Copies both grid states into a DenseGrid of the full domain size. Inactive tiles
become zero. Meant for export and debugging, not for every time step.
"""
function to_dense(grid::SparseGrid{T}) where {T}
    state_read  = _to_dense_state(grid, grid.state_read,  grid.map_read,  grid.n_active_read)
    state_write = _to_dense_state(grid, grid.state_write, grid.map_write, grid.n_active_write)
    return DenseGrid(state_read, state_write, grid.padding, grid.origin, grid.inv_dx)
end

function _to_dense_state(grid::SparseGrid{T}, state, map, n_active) where {T}
    dense     = _allocate_grid_state(CPU(), T, grid.n_nodes)
    tile_keys = Array(map.tile_keys)
    mass, wave_speed = Array(state.mass), Array(state.wave_speed)
    px, py, pz = Array(state.momentum.x), Array(state.momentum.y), Array(state.momentum.z)

    for t in 1:n_active, l in 1:TILE_VOLUME
        i, j, k = node_index(tile_keys[t], l, grid.n_tiles)
        node_in_domain(grid.n_nodes, i, j, k) || continue    # last tile may stick out

        dense.mass[i, j, k]       = mass[l, t]
        dense.wave_speed[i, j, k] = wave_speed[l, t]
        dense.momentum.x[i, j, k] = px[l, t]
        dense.momentum.y[i, j, k] = py[l, t]
        dense.momentum.z[i, j, k] = pz[l, t]
    end
    return dense
end
