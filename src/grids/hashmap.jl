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

"""
    count_limit(map::TileHashMap)

At most this many hash slots are claimed (load factor 0.75). Once the counter
reaches it, further new keys are dropped without being counted, which keeps
probe chains short and the counter meaningful. Since `capacity >= 2 max_tiles`,
the limit is always above `max_tiles`, so normal operation is never affected.
"""
@inline count_limit(map::TileHashMap) = (3 * capacity(map)) ÷ 4


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

The counter `n_active` is incremented exactly once per claimed hash slot, i.e.
it counts *distinct* tiles, no matter how many particles insert the same key.

Overflow handling (detected on the host via `tile_count`):
  - tile pool full:  the counter exceeds `max_tiles(map)`, the slot gets value 0.
                     The count stays exact, so the host can size the new pool.
  - hash table at `count_limit(map)`: new keys are dropped and not counted.
                     `tile_count(map) >= count_limit(map)` then only is a lower
                     bound and the host grows the map before sizing the pool.
"""
@inline function insert_tile!(map::TileHashMap, key::TileKey)
    mask = (capacity(map) - 1) % UInt64
    slot = hash_key(key) & mask     # <=> hash(key) mod capacity

    for _ in 1:capacity(map)
        idx = Int(slot) + 1
        current = Atomix.@atomic :monotonic map.keys[idx]   # atomic read to prevent compiler from optimizing away the loop
        current == key && return nothing                    # already registered

        if current == EMPTY_KEY
            # key is new: drop it if the table is saturated (host regrows the map)
            Atomix.@atomic(:monotonic, map.n_active[1]) >= count_limit(map) && return nothing

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

    # Hash table completely full. Only reachable when many threads pass the
    # count_limit check at the same time; then n_active == capacity >= count_limit
    # already signals saturation. Do NOT count this attempt: the counter must
    # count tiles, not insertion attempts.
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