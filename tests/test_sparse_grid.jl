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

@testset "TileHashMap counts tiles, not insertion attempts" begin
    # More distinct keys than the table can hold, each inserted many times
    map  = TileHashMap(CPU(), 16)                      # capacity 32, count limit 24
    keys = vec([tile_key(ti, tj, 0) for ti in 0:9, tj in 0:9])    # 100 keys
    for _ in 1:20, key in keys
        insert_tile!(map, key)
    end
    n = tile_count(map)
    @test n == SmashMPM.count_limit(map)               # saturated, but not inflated by repeats
    @test n <= SmashMPM.capacity(map)
    # registered keys still resolve to valid, unique tiles (or 0 if the pool was full)
    found = filter(>(0), [find_tile(map, key) for key in keys])
    @test sort(found) == collect(1:SmashMPM.max_tiles(map))
end

@testset "SparseGrid grows the pool to the needed size" begin
    T = Float64
    dx = one(T)
    spline = QuadraticSpline()
    # filled sphere of particles, many particles per tile, far more tiles than the
    # initial hash table can hold (the case that used to inflate the pool ~30x)
    pts = SVector{3, T}[]
    for i in -24:24, j in -24:24, k in -24:24
        p = SVector{3, T}(i, j, k) .* 2dx
        sum(abs2, p) <= (48dx)^2 && push!(pts, p)
    end
    pos_sa = StructArray{SVector{3, T}}((x=[p[1] for p in pts], y=[p[2] for p in pts], z=[p[3] for p in pts]))
    sets = ((particles=(pos=pos_sa,),),)

    exact = SmashMPM.count_tiles(pts, dx, zero(SVector{3, T}), spline)
    grid  = SparseGrid(dx, zero(SVector{3, T}); max_tiles=64)
    @test exact > 2 * SmashMPM.capacity(grid.map_write)       # hash table overflows

    n = activate_tiles!(grid, sets, spline)
    pool = size(grid.state_write.mass, 2)
    @test n == exact
    @test missed_count(grid.map_write) == 0
    @test exact <= pool <= ceil(Int, 1.25 * exact)              # headroom only, no blow-up
    @test SmashMPM.max_tiles(grid.map_write) == pool            # map and pool stay consistent

    # a second activation on the same grid (after a reset) needs no further growth
    grid_reset!(grid)
    grid_reset!(grid)
    @test activate_tiles!(grid, sets, spline) == exact
    @test size(grid.state_write.mass, 2) == pool
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

@testset "External forces on SparseGrid" begin
    T = Float64
    dt = 0.1
    grid = SparseGrid(1.0, zero(SVector{3, T}))
    pos = StructArray{SVector{3, T}}((x=[-3.2], y=[0.4], z=[7.9]))
    n = activate_tiles!(grid, ((particles=(pos=pos,),),), QuadraticSpline())
    fill!(view(grid.state_write.mass, :, 1:n), 2.0)

    # Constant gravity: Δp = m g dt on every active node
    apply_external_forces!(ConstantGravity(SVector{3, T}(0.0, -9.81, 0.0)), grid, dt)
    @test all(view(grid.state_write.momentum.y, :, 1:n) .≈ -1.962)
    @test all(view(grid.state_write.momentum.x, :, 1:n) .== 0.0)

    # Radial field: node on the center gets no force, its -x neighbour is pulled to +x
    fill!(grid.state_write.momentum.x, 0.0)
    center_node = (-3, 1, 8)          # global node index, position (-4, 0, 7)
    center = SmashMPM.node_position(grid.origin, grid.inv_dx, center_node...)
    apply_external_forces!(RadialInvSquareForceField(-10.0, center), grid, dt)
    l, t = find_node(grid.map_write, grid.domain, center_node...)
    @test grid.state_write.momentum.x[l, t] ≈ 0.0 atol = 1e-12
    l, t = find_node(grid.map_write, grid.domain, center_node[1] - 1, center_node[2], center_node[3])
    @test t > 0
    @test grid.state_write.momentum.x[l, t] > 0.0

    @test apply_external_forces!(NoExternalForce(), grid, dt) === nothing
    @test_throws ErrorException apply_external_forces!(VectorFieldForce(zeros(SVector{3, T}, 2, 2, 2)), grid, dt)
end

@testset "build_mpm_model with SparseGrid" begin
    T = Float64
    mat = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
    body1 = Body(Sphere(0.3, SVector(-1.0, 0.0, 0.0)), SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 2.0), mat)
    body2 = Body(Sphere(0.3, SVector( 1.0, 0.5, 0.0)), SVector(-1.0, 0.0, 0.0), zero(SVector{3, T}), mat)
    bodies = (body1, body2)

    dense_model = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0))
    open_model  = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0, grid_type=SparseGrid))
    box_model   = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0, grid_type=SparseGrid,
                                                         boundary_condition=NoSlipBoundary()))

    @test open_model.grid isa SparseGrid
    @test open_model.grid.domain isa OpenDomain
    @test box_model.grid.domain isa BoxDomain
    @test_throws ErrorException build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0, grid_type=SparseGrid,
                                                domain=OpenDomain(), boundary_condition=NoSlipBoundary()))

    ref = dense_model.grid.state_read
    for model in (open_model, box_model)
        g = model.grid
        @test g.n_active_read > 0
        @test g.n_active_write > 0          # tiles for the first g2p2g are already active
        @test missed_count(g.map_read) == 0
        d = to_dense(g).state_read
        # identical scatter -> identical totals
        @test sum(d.mass)       ≈ sum(ref.mass)       rtol = 1e-12
        @test sum(d.momentum.x) ≈ sum(ref.momentum.x) atol = 1e-10
        @test sum(d.momentum.y) ≈ sum(ref.momentum.y) atol = 1e-10
        @test maximum(d.wave_speed) ≈ maximum(ref.wave_speed)
        @test SmashMPM.max_wavespeed(g) ≈ SmashMPM.max_wavespeed(dense_model.grid)
        # box: same nodes as the dense grid
        if g.domain isa BoxDomain
            @test size(d.mass) == size(ref.mass)
            @test d.mass ≈ ref.mass
        end
    end
end

@testset "g2p2g!: SparseGrid reproduces DenseGrid" begin
    T = Float64
    mat = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
    bodies = (
        Body(Sphere(0.3, SVector(-0.4, 0.0, 0.0)), SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 2.0), mat),
        Body(Sphere(0.3, SVector( 0.4, 0.1, 0.0)), SVector(-1.0, 0.0, 0.0), zero(SVector{3, T}), mat),
    )
    sparse_setup = SimulationSetup(dx=0.1, t_max=1.0, grid_type=SparseGrid, domain=FittedBox())
    close_to(a, b) = maximum(abs.(a .- b)) <= 1e-9 * max(1.0, maximum(abs.(b)))

    @testset "single step, nodal states" begin
        dense  = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0))
        sparse = build_mpm_model(bodies, sparse_setup)
        dt = 1e-5

        SmashMPM.g2p2g!(dense, dt)
        SmashMPM.g2p2g!(sparse, dt)
        @test missed_count(sparse.grid.map_write) == 0

        d = to_dense(sparse.grid).state_write
        r = dense.grid.state_write
        @test close_to(d.mass, r.mass)
        @test close_to(d.momentum.x, r.momentum.x)
        @test close_to(d.momentum.y, r.momentum.y)
        @test close_to(d.momentum.z, r.momentum.z)
        @test close_to(d.wave_speed, r.wave_speed)

        for (ps_d, ps_s) in zip(dense.particle_sets, sparse.particle_sets)
            @test close_to(ps_s.particles.pos.x, ps_d.particles.pos.x)
            @test close_to(ps_s.particles.pos.y, ps_d.particles.pos.y)
            @test close_to(ps_s.particles.pos.z, ps_d.particles.pos.z)
            @test maximum(norm.(ps_s.particles.F .- ps_d.particles.F)) <= 1e-9
        end
    end

    @testset "5 steps with step! (reset + tile activation)" begin
        dense  = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0))
        sparse = build_mpm_model(bodies, sparse_setup)

        for _ in 1:5
            dt_d = SmashMPM.step!(dense)
            dt_s = SmashMPM.step!(sparse)
            @test dt_d ≈ dt_s
        end

        for (ps_d, ps_s) in zip(dense.particle_sets, sparse.particle_sets)
            @test close_to(ps_s.particles.pos.x, ps_d.particles.pos.x)
            @test close_to(ps_s.particles.pos.y, ps_d.particles.pos.y)
            @test close_to(ps_s.particles.pos.z, ps_d.particles.pos.z)
        end
        @test sum(to_dense(sparse.grid).state_read.mass) ≈ sum(dense.grid.state_read.mass)
    end

    @testset "export: extract_velocities and HDF5 output" begin
        dense = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0))
        open  = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0, grid_type=SparseGrid))
        box   = build_mpm_model(bodies, sparse_setup)
        for m in (dense, open, box), _ in 1:3
            SmashMPM.step!(m)
        end
        for sparse in (open, box), (ps_d, ps_s) in zip(dense.particle_sets, sparse.particle_sets)
            v_d, B_d = SmashMPM.extract_velocities(dense.grid, ps_d, dense.shapefunction)
            v_s, B_s = SmashMPM.extract_velocities(sparse.grid, ps_s, sparse.shapefunction)
            @test maximum(norm.(v_s .- v_d)) <= 1e-9 * maximum(norm.(v_d))
            @test maximum(norm.(B_s .- B_d)) <= 1e-9 * maximum(norm.(B_d))
        end

        # complete HDF5 snapshot incl. the (densified) grid
        dir = mktempdir()
        exporter = HDF5Exporter(output_dir=dir, filename_prefix="sparse_", export_grid=true)
        path = write_output(exporter, box, 3)
        @test isfile(path)
        @test isfile(joinpath(dir, "sparse_000003.xmf"))
        rm(dir; recursive=true)
    end

    @testset "OpenDomain gives the same particle motion" begin
        dense = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0))
        open  = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0, grid_type=SparseGrid))
        @test open.grid.domain isa OpenDomain
        for _ in 1:5
            SmashMPM.step!(dense)
            SmashMPM.step!(open)
        end
        for (ps_d, ps_s) in zip(dense.particle_sets, open.particle_sets)
            @test close_to(ps_s.particles.pos.x, ps_d.particles.pos.x)
        end
    end
end