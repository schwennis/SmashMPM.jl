# ---------------------------------------------------------------------------- #
#                 Sparse grid: domain-free tile keys & open domains             #
# ---------------------------------------------------------------------------- #
using .SmashMPM: tile_key, tile_coords, tile_coord, local_coord, local_index, node_index,
                find_node, insert_tile!, find_tile, tile_count, missed_count,
                TILE_SIZE, TILE_VOLUME, TILE_COORD_MIN, TILE_COORD_MAX, EMPTY_KEY

@testset "Tile keys (signed, domain independent)" begin
    # floor semantics for negative node indices
    @test tile_coord(1)  == 0
    @test tile_coord(TILE_SIZE) == 0
    @test tile_coord(TILE_SIZE + 1) == 1
    @test tile_coord(0)  == -1
    @test tile_coord(-TILE_SIZE + 1) == -1
    @test tile_coord(-TILE_SIZE) == -2
    @test local_coord(0) == TILE_SIZE - 1
    @test local_coord(1) == 0

    # key round trip incl. the extreme coordinates
    for (ti, tj, tk) in ((0, 0, 0), (-1, 2, -3), (TILE_COORD_MIN, TILE_COORD_MAX, 0),
                         (TILE_COORD_MAX, TILE_COORD_MIN, TILE_COORD_MAX))
        key = tile_key(ti, tj, tk)
        @test key != EMPTY_KEY
        @test tile_coords(key) == (ti, tj, tk)
    end
    @test tile_key(1, 0, 0) != tile_key(0, 1, 0) != tile_key(0, 0, 1)

    # node <-> (key, l) round trip, also for negative indices
    for (i, j, k) in ((1, 1, 1), (0, -5, 7), (-17, 3, -1), (123, -456, 789))
        key = tile_key(tile_coord(i), tile_coord(j), tile_coord(k))
        @test node_index(key, local_index(i, j, k)) == (i, j, k)
    end
end

@testset "TileHashMap insert / find" begin
    map = TileHashMap(CPU(), 64)
    keys = vec([tile_key(ti, tj, tk) for ti in -2:1, tj in -1:1, tk in 0:1])
    for key in keys
        insert_tile!(map, key)
        insert_tile!(map, key)              # duplicates are ignored
    end
    @test tile_count(map) == length(keys)
    tiles = [find_tile(map, key) for key in keys]
    @test sort(tiles) == collect(1:length(keys))
    @test find_tile(map, tile_key(100, 100, 100)) == 0
    @test all(map.tile_keys[t] == keys[n] for (n, t) in enumerate(tiles))
end

@testset "SparseGrid with OpenDomain" begin
    T = Float64
    dx = 0.5
    grid = SparseGrid(dx, zero(SVector{3, T}); max_tiles=2)    # tiny pool -> must grow
    @test grid.domain isa OpenDomain
    spline = QuadraticSpline()

    # particles far on the negative side and far away from each other
    pos = [SVector{3, T}(-37.3, -0.2, 5.1), SVector{3, T}(1.0e4, -2.0e3, 0.0)]
    pos_sa = StructArray{SVector{3, T}}((x=[p[1] for p in pos], y=[p[2] for p in pos], z=[p[3] for p in pos]))
    p_set = (particles=(pos=pos_sa,),)

    n = activate_tiles!(grid, (p_set,), spline)
    @test n == grid.n_active_write
    @test n == SmashMPM.count_tiles(pos, dx, zero(SVector{3, T}), spline)
    @test SmashMPM.max_tiles(grid.map_write) >= n            # pool has grown
    @test missed_count(grid.map_write) == 0

    # every stencil node of every particle is found
    for p in pos
        gp = SmashMPM.grid_position(p, inv(dx), zero(SVector{3, T}))
        base = SmashMPM.get_support_base(spline, gp)
        for di in 0:2, dj in 0:2, dk in 0:2
            _, t = find_node(grid.map_write, grid.domain, base[1] + di, base[2] + dj, base[3] + dk)
            @test t > 0
        end
    end

    # far apart particles: the dense bounding box would be huge -> refused
    @test_throws ErrorException to_dense(grid)

    # write some mass and check to_dense conserves it (only the nearby particle)
    grid_reset!(grid)
    grid_reset!(grid)
    near = StructArray{SVector{3, T}}((x=[-37.3, -30.0], y=[-0.2, 1.0], z=[5.1, 4.0]))
    n = activate_tiles!(grid, ((particles=(pos=near,),),), spline)
    grid.state_write.mass[1, 1] = 2.0
    grid.state_write.mass[TILE_VOLUME, n] = 3.0
    dense = to_dense(grid)
    @test sum(dense.state_write.mass) ≈ 5.0

    # NaN particles are counted as missed instead of crashing
    grid_reset!(grid)
    bad = StructArray{SVector{3, T}}((x=[NaN], y=[0.0], z=[0.0]))
    activate_tiles!(grid, ((particles=(pos=bad,),),), spline)
    @test missed_count(grid.map_write) == 1
    @test grid.n_active_write == 0

    # walls need a box
    @test_throws ErrorException apply_boundary_condition!(NoSlipBoundary(), grid)
    @test apply_boundary_condition!(NullBoundaryCondition(), grid) === nothing
end

@testset "SparseGrid with BoxDomain matches DenseGrid nodes" begin
    T = Float64
    N = SVector(10, 10, 10)
    grid = SparseGrid(1.0, N, zero(SVector{3, T}), 2)
    @test grid.domain == BoxDomain(N)

    # particle near the corner: stencil gets clipped to the box
    pos_sa = StructArray{SVector{3, T}}((x=[0.1], y=[0.1], z=[0.1]))
    n = activate_tiles!(grid, ((particles=(pos=pos_sa,),),), QuadraticSpline())
    @test n == 1
    _, t = find_node(grid.map_write, grid.domain, 0, 1, 1)
    @test t == 0                                   # outside the box

    grid.state_write.momentum.x[:, 1] .= 1.0
    apply_boundary_condition!(NoSlipBoundary(), grid)
    l, t = find_node(grid.map_write, grid.domain, 3, 3, 3)
    @test grid.state_write.momentum.x[l, t] == 1.0 # interior
    l, t = find_node(grid.map_write, grid.domain, 2, 3, 3)
    @test grid.state_write.momentum.x[l, t] == 0.0 # padding
end