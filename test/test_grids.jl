# ---------------------------------------------------------------------------- #
#       Grid infrastructure: tile keys, hash map, activation, grid operations   #
# ---------------------------------------------------------------------------- #
# The SparseGrid must behave exactly like a DenseGrid on the same nodes. Every
# grid operation (boundary condition, external forces, kick, reset) is applied
# to a DenseGrid and to an identically filled SparseGrid(BoxDomain); the results
# must agree node by node AND match the analytic expectation.
using .SmashMPM: tile_key, tile_coords, tile_coord, local_index, node_index, find_node,
                insert_tile!, find_tile, tile_count, missed_count

@testset "Tile keys" begin
    # floor semantics for node indices ≤ 0
    T4 = S.TILE_SIZE
    @test tile_coord.((-T4, -T4 + 1, 0, 1, T4, T4 + 1)) == (-2, -1, -1, 0, 0, 1)

    # node -> (key, l) -> node, random indices of both signs
    rng = Xoshiro(7)
    @test all(1:1000) do _
        i, j, k = rand(rng, -10^6:10^6, 3)
        node_index(tile_key(tile_coord(i), tile_coord(j), tile_coord(k)), local_index(i, j, k)) == (i, j, k)
    end

    # key round trip at the limits of the key range; no key equals EMPTY_KEY; no collisions
    ext = (S.TILE_COORD_MIN, S.TILE_COORD_MAX, 0, -1)
    @test all(Iterators.product(ext, ext, ext)) do (a, b, c)
        key = tile_key(a, b, c)
        key != S.EMPTY_KEY && tile_coords(key) == (a, b, c)
    end
    @test allunique(tile_key(a, b, c) for a in -3:3, b in -3:3, c in -3:3)
end

@testset "TileHashMap" begin
    map  = TileHashMap(CPU(), 64)
    keys = vec([tile_key(ti, tj, tk) for ti in -2:1, tj in -1:1, tk in 0:1])
    for key in keys, _ in 1:3                              # duplicates are ignored
        insert_tile!(map, key)
    end
    tiles = find_tile.(Ref(map), keys)
    @test tile_count(map) == length(keys)
    @test sort(tiles) == 1:length(keys)                    # dense, unique tile indices
    @test map.tile_keys[tiles] == keys                     # reverse map tile -> key
    @test find_tile(map, tile_key(100, 100, 100)) == 0

    # saturated table: counts tiles (not insertion attempts), found tiles stay valid
    small = TileHashMap(CPU(), 16)
    many  = vec([tile_key(ti, tj, 0) for ti in 0:9, tj in 0:9])
    for _ in 1:20, key in many
        insert_tile!(small, key)
    end
    @test tile_count(small) == S.count_limit(small)
    @test sort(filter(>(0), find_tile.(Ref(small), many))) == 1:S.max_tiles(small)
end

@testset "SparseGrid tile activation" begin
    spline = QuadraticSpline()
    o      = zero(V3)

    # filled ball with far more tiles than the initial pool / hash table: grows once, no blow-up
    pts   = [2.0 * V3(i, j, k) for i in -24:24, j in -24:24, k in -24:24 if i^2 + j^2 + k^2 <= 24^2]
    exact = S.count_tiles(pts, 1.0, o, spline)
    grid  = SparseGrid(1.0, o; max_tiles = 64)
    # (the pool growth is intended here, its @info would only be noise)
    @test with_logger(() -> activate_tiles!(grid, (position_set(pts),), spline), NullLogger()) == exact
    pool  = S.pool_tiles(grid)
    @test missed_count(grid.map_write) == 0
    @test exact <= pool <= ceil(Int, 1.25 * exact) && S.max_tiles(grid.map_write) == pool
    grid_reset!(grid); grid_reset!(grid)
    @test activate_tiles!(grid, (position_set(pts),), spline) == exact && S.pool_tiles(grid) == pool

    # open domain: every stencil node (+ margin) of far away / negative particles is active
    far  = [V3(-37.3, -0.2, 5.1), V3(1.0e4, -2.0e3, 0.0), V3(-5e5, 3e5, -1e5)]
    grid = SparseGrid(0.5, o; max_tiles = 2)
    with_logger(() -> activate_tiles!(grid, (position_set(far),), spline), NullLogger())
    @test all(far) do x
        lo, hi = S.stencil_node_range(spline, S.grid_position(x, grid.inv_dx, o), S.ACTIVATION_MARGIN)
        all(find_node(grid.map_write, grid.domain, i, j, k)[2] > 0
            for i in lo[1]:hi[1], j in lo[2]:hi[2], k in lo[3]:hi[3])
    end
    @test_throws ErrorException to_dense(grid)            # widely separated: dense box refused

    # NaN and out-of-range particles are counted as lost instead of crashing
    grid_reset!(grid)
    activate_tiles!(grid, (position_set([V3(NaN, 0, 0), V3(1e30, 0, 0)]),), spline)
    @test missed_count(grid.map_write) == 2 && grid.n_active_write == 0
end


# ---------------------------------------------------------------------------- #
#                  Grid operations: DenseGrid vs SparseGrid(Box)               #
# ---------------------------------------------------------------------------- #
const TWIN_N      = SVector(10, 9, 8)
const TWIN_DX     = 0.5
const TWIN_ORIGIN = V3(-1.0, 0.5, 2.0)

"DenseGrid and SparseGrid(BoxDomain) on the same nodes, write states filled with the same random data."
function twin_grids(rng; padding = 2)
    dense  = DenseGrid(TWIN_DX, TWIN_N, TWIN_ORIGIN, padding)
    sparse = SparseGrid(TWIN_DX, TWIN_N, TWIN_ORIGIN, padding)
    nodes  = vec([S.node_position(TWIN_ORIGIN, 1 / TWIN_DX, Tuple(I)...) for I in CartesianIndices(Tuple(TWIN_N))])
    activate_tiles!(sparse, (position_set(nodes),), QuadraticSpline())       # all tiles of the box
    for I in CartesianIndices(dense.state_write.mass)
        m    = rand(rng) < 0.2 ? 0.0 : rand(rng)                             # some empty nodes
        node = GridNode(mass = m, momentum = V3(randn(rng, 3)), force = V3(randn(rng, 3)), e_mix = 0.0)
        dense.state_write[I] = node
        l, t = find_node(sparse.map_write, sparse.domain, Tuple(I)...)
        sparse.state_write[l, t] = node
    end
    return dense, sparse
end

"Largest difference of the write states on the nodes of the box."
function twin_diff(dense, sparse)
    d, s, err = dense.state_write, sparse.state_write, 0.0
    for I in CartesianIndices(d.mass)
        l, t = find_node(sparse.map_write, sparse.domain, Tuple(I)...)
        err  = max(err, abs(d.mass[I] - s.mass[l, t]), norm(d.momentum[I] - s.momentum[l, t]),
                   norm(d.force[I] - s.force[l, t]))
    end
    return err
end

@testset "Grid operations: analytic result, SparseGrid == DenseGrid" begin
    dt     = 0.1
    g      = V3(0, -9.81, 1)
    center = S.node_position(TWIN_ORIGIN, 1 / TWIN_DX, 5, 5, 4)             # source sits on a node
    F0     = -3.0
    field  = [V3(sin(i), cos(j), i * j * k / 100) for i in 1:TWIN_N[1], j in 1:TWIN_N[2], k in 1:TWIN_N[3]]
    pad(I) = any(d -> I[d] <= 2 || I[d] > TWIN_N[d] - 2, 1:3)
    radial(x) = (r = x - center; F0 * r / (norm(r) + eps())^3)

    # (operation, expected (momentum, force) of a node from its old state)
    cases = (
        "NoSlipBoundary"   => (gr -> apply_boundary_condition!(NoSlipBoundary(), gr),
                               (n, x, I) -> (pad(I) ? zero(V3) : n.momentum, n.force)),
        "ConstantGravity"  => (gr -> apply_external_forces!(ConstantGravity(g), gr, dt),
                               (n, x, I) -> (n.momentum, n.force + n.mass * g)),
        "RadialInvSquare"  => (gr -> apply_external_forces!(RadialInvSquareForceField(F0, center), gr, dt),
                               (n, x, I) -> (n.momentum, n.force + (n.mass > eps() ? n.mass * radial(x) : zero(V3)))),
        "VectorFieldForce" => (gr -> apply_external_forces!(VectorFieldForce(field), gr, dt),
                               (n, x, I) -> (n.momentum, n.force + (n.mass > eps() ? n.mass * field[I] : zero(V3)))),
        "grid_kick!"       => (gr -> S.grid_kick!(gr, dt),
                               (n, x, I) -> (n.momentum + dt * n.force, n.force)),
    )
    rng = Xoshiro(11)
    @testset "$name" for (name, (op!, expected)) in cases
        dense, sparse = twin_grids(rng)
        before = copy(dense.state_write)
        op!(dense)
        op!(sparse)
        err = 0.0
        for I in CartesianIndices(before.mass)
            x = S.node_position(TWIN_ORIGIN, 1 / TWIN_DX, Tuple(I)...)
            p, f = expected(before[I], x, I)
            err = max(err, norm(dense.state_write.momentum[I] - p), norm(dense.state_write.force[I] - f),
                      abs(dense.state_write.mass[I] - before.mass[I]))
        end
        @test err < 1e-12                                  # analytic
        @test twin_diff(dense, sparse) < 1e-14             # sparse kernels = dense kernels
    end

    # walls and bounded fields need a box
    open = SparseGrid(1.0, zero(V3))
    @test_throws ErrorException apply_boundary_condition!(NoSlipBoundary(), open)
    @test_throws ErrorException apply_external_forces!(VectorFieldForce(field), open, dt)
    @test apply_boundary_condition!(NullBoundaryCondition(), open) === nothing
end

@testset "Grid reset and to_dense" begin
    dense, sparse = twin_grids(Xoshiro(2))
    for grid in (dense, sparse)
        written = grid.state_write
        grid_reset!(grid)
        @test grid.state_read === written                  # swap, no copy
        @test all(f -> all(iszero, f), (grid.state_write.mass, grid.state_write.momentum.x,
                                        grid.state_write.force.z, grid.state_write.e_mix))
    end
    @test sparse.n_active_write == 0 && sparse.n_active_read > 0

    # the dense copy of the sparse read state is exactly the dense read state
    d = to_dense(sparse)
    @test d.origin ≈ dense.origin
    @test d.state_read.mass == dense.state_read.mass
    @test d.state_read.momentum == dense.state_read.momentum
end
