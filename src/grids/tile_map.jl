# ---------------------------------------------------------------------------- #
#                                   Tile Map                                   #
# ---------------------------------------------------------------------------- #
# A tile map is a subtype of
abstract type AbstractTileMap end
# and is responsible to map between tile keys and tile indices in the
# tile pool of a SparseGrid.
#
# Tile keys (TileKey = UInt64, see tile_keys.jl) encode signed tile coordinates
# and are independent of any domain size, so a tile map can be used for
# unbounded domains (open boundary conditions).
#
# It must have the fields
#   - tile_keys: Vector containing mapping tile index -> key
#                   Its length is max_tiles
#   - n_active: one-element vector containing number of activated tiles
#   - n_missed: one-element vector counting lost writes / lost particles.
#               Should be zero.
#
# The one-element vectors are used to enable atomic operations
#
# The following functions must be implemented for each concrete subtype:
#   - new_tile_map(type(map), backend, max_tiles) -> map
#   - insert_tile!(map, key::TileKey)
#   - find_tile(map, key::TileKey) -> tile_index (0 if not registered)
#   - clear!(map) -> nothing
# The Struct must be adapted with Adapt.jl to enable backend compatibility:
# Use Adapt.@adapt_structure TileMapType
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                   Interface                                  #
# ---------------------------------------------------------------------------- #
max_tiles(map::AbstractTileMap) = length(map.tile_keys)

# Growing the sparse grid requires allocation of new arrays which must be done on host
tile_count(map::AbstractTileMap)   = Int(Array(map.n_active)[1])
missed_count(map::AbstractTileMap) = Int(Array(map.n_missed)[1])

"""
    record_miss!(map)
Count one lost write (P2G into an inactive tile) or one lost particle
(position not representable by tile keys). Device-safe.
"""
@inline function record_miss!(map::AbstractTileMap)
    Atomix.@atomic map.n_missed[1] += Int32(1)
    return nothing
end

new_tile_map(::Type{M}, backend, max_tiles::Int) where {M<:AbstractTileMap} = _not_implemented(:new_tile_map, M)
insert_tile!(map::AbstractTileMap, key) = _not_implemented(:insert_tile!, map)
find_tile(map::AbstractTileMap, key) = _not_implemented(:find_tile, map)
clear!(map::AbstractTileMap) = _not_implemented(:clear!, map)


# ---------------------------------------------------------------------------- #
#                                   Includes                                   #
# ---------------------------------------------------------------------------- #
include("hashmap.jl")
