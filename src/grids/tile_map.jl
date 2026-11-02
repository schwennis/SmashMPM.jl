# ---------------------------------------------------------------------------- #
#                                   Tile Map                                   #
# ---------------------------------------------------------------------------- #
# A tile map is a subtype of 
abstract type AbstractTileMap end
# is responsible to map between positions and tile indices in the 
# tile pool of a SparseGrid
#
# It must have the fields
#   - tile_keys: Vector containing mapping tile index <-> key
#                   Its length is max_tiles
#   - n_active: one-element vector containing number of activated tiles
#   - n_missed: one-element vector containing number of missed tiles. Should be zero.
#
# The one-element vectors are used to enable atomic operations
#
# The following functions must be implemented for each concrete subtype:
#   - new_tile_map(type(map), backend, max_tiles) -> map
#   - insert_tile!(map, key)
#   - find_tile(map, key) -> tile_index
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


insert_tile!(map::AbstractTileMap, key) = _not_implemented(:insert_tile!, map)
find_tile(map::AbstractTileMap, key) = _not_implemented(:find_tile, map)
clear!(map::AbstractTileMap) = _not_implemented(:clear, map)





# ---------------------------------------------------------------------------- #
#                                   Includes                                   #
# ---------------------------------------------------------------------------- #
include("hashmap.jl")