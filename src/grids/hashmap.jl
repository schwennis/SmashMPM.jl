# ---------------------------------------------------------------------------- #
#                                    Hashmap                                   #
# ---------------------------------------------------------------------------- #
# Maps a tile key (TileKey = UInt64, see tile_keys.jl) to a tile index.
# Tile indices are dense in 1:n_active.
#
# Tile hashing as in Qiu et al. 2023:
# Open addressing with linear probing
#   - insert:   atomic compare-and-swap on the key slot.
#               winning thread takes next free tile index
#               from an atomic counter
#   - find:     Uses hash function to find the slot
#               corresponding to the key, then linear probing
#               until the key is found or an empty slot is reached
#
# Insertion and looking up are done in separate kernel launches.
#
# The keys encode signed tile coordinates and do NOT depend on a domain size,
# so the hash map works for open boundaries (unbounded domains).
# Empty slots hold EMPTY_KEY (typemax(UInt64), never a valid key).
# ---------------------------------------------------------------------------- #
struct TileHashMap{K, V, O} <: AbstractTileMap
    keys::K         # hash slots containing tile keys or EMPTY_KEY (UInt64)
    values::V       # hash slot -> tile index (Int32), 0 = pool was full
    tile_keys::K    # reverse map: tile index -> tile key (UInt64)
    n_active::O     # one-element array: number of activated tiles (+ overflow signal)
    n_missed::O     # one-element array: number of lost writes / particles. Should be zero
end

Adapt.@adapt_structure TileHashMap  # Backend Compatibility


function TileHashMap(backend, max_tiles::Int)
    capacity = nextpow(2, 2 * max_tiles)    # load factor <= 0.5 keeps probe chains short
    return TileHashMap(
        _to_backend(backend, fill(EMPTY_KEY, capacity)),    # UInt64
        _to_backend(backend, zeros(Int32, capacity)),
        _to_backend(backend, fill(EMPTY_KEY, max_tiles)),   # UInt64
        _to_backend(backend, zeros(Int32, 1)),
        _to_backend(backend, zeros(Int32, 1)),
    )
end


@inline new_tile_map(::Type{<:TileHashMap}, backend, max_tiles::Int) = TileHashMap(backend, max_tiles)


# helpers
@inline capacity(map::TileHashMap) = length(map.keys)


function clear!(map::TileHashMap)
    fill!(map.keys, EMPTY_KEY)
    fill!(map.n_active, zero(Int32))
    fill!(map.n_missed, zero(Int32))
    return nothing
end


"""
    hash_key(k::UInt64)
Hash function for tile keys. Uses the 64-bit MurmurHash3 finalizer (fmix64).
Mixes all three packed coordinate fields into the low bits used for the slot.
"""
@inline function hash_key(k::UInt64)
    k ⊻= k >> 33
    k *= 0xff51afd7ed558ccd
    k ⊻= k >> 33
    k *= 0xc4ceb9fe1a85ec53
    k ⊻= k >> 33
    return k
end


"""
    insert_tile!(map, key)

Register `key` in the map (no-op if it is already there). Safe to call from
many threads at once.

Overflow handling (both are detected on the host via `tile_count`):
  - tile pool full:  the counter exceeds `max_tiles(map)`, the slot gets value 0
  - hash table full: the counter is still incremented, so that
                     `tile_count(map) > max_tiles(map)` and the host grows the pool
"""
@inline function insert_tile!(map::TileHashMap, key::TileKey)
    mask = (capacity(map) - 1) % UInt64
    slot = hash_key(key) & mask     # <=> hash(key) mod capacity

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

        slot = (slot + one(UInt64)) & mask              # linear probing
    end

    # Hash table completely full (cannot happen while n_active <= max_tiles,
    # since capacity >= 2 max_tiles). Signal overflow so the host grows the map.
    Atomix.@atomic map.n_active[1] += Int32(1)
    return nothing
end


"""
    find_tile(map, key) -> tile index
Returns tile index or 0 if the tile is not registered
"""
@inline function find_tile(map::TileHashMap, key::TileKey)
    mask = (capacity(map) - 1) % UInt64
    slot = hash_key(key) & mask     # <=> hash(key) mod capacity

    for _ in 1:capacity(map)
        idx = Int(slot) + 1
        current = map.keys[idx]
        current == key       && return Int(map.values[idx])
        current == EMPTY_KEY && return 0
        slot = (slot + one(UInt64)) & mask
    end
    return 0
end
