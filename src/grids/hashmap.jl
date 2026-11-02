# ---------------------------------------------------------------------------- #
#                                    Hashmap                                   #
# ---------------------------------------------------------------------------- #
# Maps a tile key (UInt32) to a tile index
# Tile indices are dense in 1:n_tiles
#
# Algorithm after after Qiu et al. 2023:
# Open addressing with linear probing
#   - insert:   atomic compare-and-swap on the key slot.
#               winning thread takes next free tile index 
#               from an atomic counter
#   - find:     Uses hash function to find the slot 
#               corresponding to the key, then linear probing 
#               until the key is found or an empty slot is reached
#
#
# Insertion and looking up are done in separate kernel launches
#
# Non-activated indices use type max as value:
const EMPTY_KEY = typemax(UInt32)
#
# Use this for open boundary conditions since the domain size 
# is never specified.
# ----------------------------------------------------------------------------
struct TileHashMap{K, V, O} <: AbstractTileMap
    keys::K         # hash slots containing tile keys or EMPTY_KEY (UInt32)
    values::V       # key map: tile_key -> tile index (Int32)
    tile_keys::K    # reverse map: tile index -> tile key (Uint32)
    n_active::O     # array containing only number of active tiles
    n_missed::O     # array containing only number of missed tiles. Should be zero
end

Adapt.@adapt_structure TileHashMap  # Backend Compatibility


function TileHashMap(backend, max_tiles::Int)
    capacity = nextpow(2, 2 * max_tiles)    # load factor <= 0.5 keeps probe chains short
    return TileHashMap(
        _to_backend(backend, fill(EMPTY_KEY, capacity)),    # UInt32
        _to_backend(backend, zeros(Int32, capacity)),
        _to_backend(backend, fill(EMPTY_KEY, max_tiles)),   # UInt32
        _to_backend(backend, zeros(Int32, 1)),
        _to_backend(backend, zeros(Int32, 1)),
    )
end


# helpers
@inline capacity(map::TileHashMap) = length(map.keys)


function clear!(map::TileHashMap)
    fill!(map.keys, EMPTY_KEY)
    fill!(map.n_tiles, zero(Int32))
    fill!(map.n_missed, zero(Int32))
end


"""
    hash_key(k::UInt32)
Hash function for tile keys. Uses MurmurHash3 finalizer.
"""
@inline function hash_key(k::UInt32)
    k ⊻= k >> 16
    k *= 0x85ebca6b
    k ⊻= k >> 13
    k *= 0xc2b2ae35
    k ⊻= k >> 16
    return k
end


"""
    insert_tile!(map, key)

Register `key` in the map (no-op if it is already there). Safe to call from
many threads at once.
"""
@inline function insert_tile!(map::TileHashMap, key::UInt32)
    mask = UInt32(capacity(map) - 1)
    slot = hash_key(key) & mask # <=> hash(key) mod capacity

    for _ in 1:capacity(map)
        idx = Int(slot) + 1
        current = Atomix.@atomic :monotonic map.keys[idx]   # atomic read to prevent compiler from optimizing away the loop
        current == key && return nothing                    # already registered

        if current == EMPTY_KEY
            res = Atomix.@atomicreplace map.keys[idx] EMPTY_KEY => key
            if res.success
                # This thread owns the slot: hand out the next tile index
                tile = Atomix.@atomic map.n_active[1] += Int32(1)
                if tile <= max_tiles(map)
                    map.values[idx] = tile
                    map.tile_keys[tile] = key
                else
                    map.values[idx] = Int32(0)          # pool full; host grows it and retries
                end
                return nothing
            end
            res.old == key && return nothing            # another thread was faster
        end

        slot = (slot + one(UInt32)) & mask              # linear probing
    end
    return nothing
end



"""
    find_tile(map, key) -> tile index
Returns tile index or 0 if the tile is not registered
"""
@inline function find_tile(map::TileHashMap, key::UInt32)
    mask = UInt32(capacity(map) - 1)
    slot = hash_key(key) & mask # <=> hash(key) mod capacity

    for _ in 1:capacity(map)
        idx = Int(slot) + 1
        current = map.keys[idx]
        current == key       && return Int(map.values[idx])
        current == EMPTY_KEY && return 0
        slot = (slot + one(UInt32)) & mask
    end
    return false    # type neutral 0
end