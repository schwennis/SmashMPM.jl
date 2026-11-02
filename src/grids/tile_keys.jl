# ---------------------------------------------------------------------------- #
#                               Tiles and tile keys                            #
# ---------------------------------------------------------------------------- #
# TILE_SIZE^3 nodes form a tile. TILE_SIZE must be a power of 2, so that the
# tile coordinate of a node is an arithmetic shift and the local coordinate is
# a bit mask. Both are floor-based and therefore also correct for node
# indices <= 0, i.e. for nodes "left of" the grid origin.
const TILE_SIZE   = 4                           # nodes per tile edge
const TILE_SHIFT  = trailing_zeros(TILE_SIZE)   # log2(TILE_SIZE)
const TILE_MASK   = TILE_SIZE - 1
const TILE_VOLUME = TILE_SIZE^3                 # nodes per tile
@assert ispow2(TILE_SIZE) "TILE_SIZE must be a power of 2"

# In each g2p2g step, the particles move at most 1 node (CFL condition).
# Since the P2G step uses the new position, the stencil of activated tiles
# must be extended by a margin in each direction:
const ACTIVATION_MARGIN = 1                     # extra nodes around the stencil
# (This margin definition differs from the implementation in Qiu et al. 2023
# to allow the g2p2g fusion)


# ---------------------------------------------------------------------------- #
#                         Domain-free tile keys (UInt64)                       #
# ---------------------------------------------------------------------------- #
# A key packs the three *signed* tile coordinates (ti, tj, tk) into one UInt64,
# 21 bits per axis, each shifted by KEY_BIAS = 2^20:
#
#   bit  0..20 : ti + KEY_BIAS
#   bit 21..41 : tj + KEY_BIAS
#   bit 42..62 : tk + KEY_BIAS
#   bit 63     : always 0
#
# The key therefore does NOT depend on the domain size (open boundaries).
# Representable tile coordinates: [-2^20, 2^20 - 1] per axis
#   = ±4.19e6 nodes per axis for TILE_SIZE = 4 (e.g. ±125 km at dx = 3 cm).
# Bit 63 is never set, so typemax(UInt64) can never be a valid key.
const TileKey        = UInt64
const KEY_BITS       = 21
const KEY_BIAS       = 1 << (KEY_BITS - 1)                      # 2^20
const KEY_FIELD_MASK = (one(TileKey) << KEY_BITS) - one(TileKey)
const EMPTY_KEY      = typemax(TileKey)                         # marks an empty hash slot
const TILE_COORD_MIN = -KEY_BIAS
const TILE_COORD_MAX = KEY_BIAS - 1


"""Tile coordinate (0-based, signed) of a 1-based node index along one axis."""
@inline tile_coord(i) = (i - 1) >> TILE_SHIFT           # == fld(i - 1, TILE_SIZE)

"""Local coordinate (0-based) of a 1-based node index inside its tile."""
@inline local_coord(i) = (i - 1) & TILE_MASK            # == mod(i - 1, TILE_SIZE)

"""Can the tile coordinates be encoded in a key without aliasing?"""
@inline tile_in_key_range(ti, tj, tk) =
    (TILE_COORD_MIN <= ti <= TILE_COORD_MAX) &
    (TILE_COORD_MIN <= tj <= TILE_COORD_MAX) &
    (TILE_COORD_MIN <= tk <= TILE_COORD_MAX)

"""
    tile_key(ti, tj, tk) -> TileKey

Encode signed tile coordinates into a key. The caller must make sure that
`tile_in_key_range(ti, tj, tk)` holds, otherwise the coordinates wrap around.
"""
@inline function tile_key(ti, tj, tk)
    ui = ((ti + KEY_BIAS) % TileKey) & KEY_FIELD_MASK
    uj = ((tj + KEY_BIAS) % TileKey) & KEY_FIELD_MASK
    uk = ((tk + KEY_BIAS) % TileKey) & KEY_FIELD_MASK
    return ui | (uj << KEY_BITS) | (uk << (2 * KEY_BITS))
end

"""
    tile_coords(key) -> (ti, tj, tk)

Decode a key into signed tile coordinates (inverse of `tile_key`).
"""
@inline function tile_coords(key::TileKey)
    ti = (key & KEY_FIELD_MASK) % Int - KEY_BIAS
    tj = ((key >> KEY_BITS) & KEY_FIELD_MASK) % Int - KEY_BIAS
    tk = ((key >> (2 * KEY_BITS)) & KEY_FIELD_MASK) % Int - KEY_BIAS
    return ti, tj, tk
end

"""Linear local index l ∈ 1:TILE_VOLUME of a node inside its tile."""
@inline local_index(i, j, k) =
    1 + local_coord(i) + TILE_SIZE * (local_coord(j) + TILE_SIZE * local_coord(k))

"""
    node_index(key, l) -> (i, j, k)

Global node indices (1-based, may be <= 0) of local node `l` in tile `key`.
"""
@inline function node_index(key::TileKey, l)
    ti, tj, tk = tile_coords(key)
    l0 = l - 1
    li = l0 & TILE_MASK
    lj = (l0 >> TILE_SHIFT) & TILE_MASK
    lk = l0 >> (2 * TILE_SHIFT)
    return 1 + TILE_SIZE * ti + li, 1 + TILE_SIZE * tj + lj, 1 + TILE_SIZE * tk + lk
end
