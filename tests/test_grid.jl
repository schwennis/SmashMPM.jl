# --- Setup ---
T = Float64
dx = 0.5
N = SVector{3, Int}(4, 4, 4)
origin = SVector{3, T}(0.0, 0.0, 0.0)
padding = 2

@testset "DenseGrid Initialization & Type Stability" begin
    grid = DenseGrid(dx, N, origin, padding, KernelAbstractions.CPU())

    @inferred DenseGrid(dx, N, origin, padding, KernelAbstractions.CPU()) # Type stability check for constructor
    @test_call DenseGrid(dx, N, origin, padding, KernelAbstractions.CPU())
    @test_opt DenseGrid(dx, N, origin, padding, KernelAbstractions.CPU())
    
    @test grid.inv_dx ≈ 1.0 / dx
    @test grid.padding == padding
    @test size(grid.state_read) == (4, 4, 4)
end

@testset "Grid Reset (In-Place Mutation)" begin
    grid = DenseGrid(dx, N, origin, padding, KernelAbstractions.CPU())
    
    # Populate dummy values into the 'new' state
    # Note: Accessing fields using correct GridNode names (mass, momentum, force)
    grid.state_write.mass[1,1,1] = 42.0
    grid.state_write.force.x[1,1,1] = 10.0
    
    # Track object IDs before reset to verify a true pointer swap occurs
    old_id_before = objectid(grid.state_read)
    new_id_before = objectid(grid.state_write)

    SmashMPM.grid_reset!(grid)

    # 1. Did the buffers swap locations?
    @test objectid(grid.state_read) == new_id_before
    @test objectid(grid.state_write) == old_id_before

    # 2. Were values retained in 'old' and cleared in 'new'?
    @test grid.state_read.mass[1,1,1] == 42.0
    @test grid.state_write.mass[1,1,1] == 0.0
    @test grid.state_read.force.x[1,1,1] == 10.0
    @test grid.state_write.force.x[1,1,1] == 0.0
end

@testset "Reductions & Grid Performance (JET)" begin
    grid = DenseGrid(dx, N, origin, padding, KernelAbstractions.CPU())

    # The CFL reduction works on the particle wave speeds (no grid field anymore).
    # Minimal particle sets: max_wavespeed only reads ps.particles.wave_speed.
    # The empty set must not throw (init = 0).
    sets = ((particles = (wave_speed = [1.0, 15.5, 2.0],),),
            (particles = (wave_speed = Float64[],),),
            (particles = (wave_speed = [3.0],),))

    # Test inference of the reduction (mapreduce is a common source of type instability)
    @test @inferred(SmashMPM.max_wavespeed(sets)) == 15.5
    @test SmashMPM.max_wavespeed(((particles = (wave_speed = Float64[],),),)) == 0.0

    # JET static analysis for the reduction and the grid operations
    @test_call SmashMPM.max_wavespeed(sets)
    @test_opt SmashMPM.max_wavespeed(sets)

    @test_call SmashMPM.grid_reset!(grid)
    @test_opt SmashMPM.grid_reset!(grid)
end