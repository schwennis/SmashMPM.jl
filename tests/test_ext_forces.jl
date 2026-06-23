
# --- Shared Grid Setup ---
T = Float64
dx = 1.0
N = SVector{3, Int}(4, 4, 4)
origin = SVector{3, T}(0.0, 0.0, 0.0)
padding = 0
dt = 0.1

grid = DenseGrid(dx, N, origin, padding, CPU())

# Helper function to reset and prime the grid with mass
function reset_grid_mass!(grid, initial_mass=2.0)
    fill!(grid.state_new.mass, initial_mass)
    fill!(grid.state_new.momentum, zero(SVector{3, T}))
end

@testset "1. No External Force" begin
    reset_grid_mass!(grid)
    force = NoExternalForce()

    @test_nowarn apply_external_forces!(force, grid, dt)
    
    # Verify momentum remains untouched (completely zero)
    @test all(grid.state_new.momentum.x .== 0.0)
    @test all(grid.state_new.momentum.y .== 0.0)
    @test all(grid.state_new.momentum.z .== 0.0)

    # Performance assertions
    @test_call apply_external_forces!(force, grid, dt)
    @test_opt apply_external_forces!(force, grid, dt)
end

@testset "2. Constant Gravity" begin
    reset_grid_mass!(grid, 2.0) # mass = 2.0
    g_vec = SVector{3, T}(0.0, -9.81, 0.0)
    force = ConstantGravity(g_vec)

    apply_external_forces!(force, grid, dt)

    # Expected momentum Δp = m * g * dt = 2.0 * -9.81 * 0.1 = -1.962
    expected_py = -1.962
    @test all(grid.state_new.momentum.x .== 0.0)
    @test all(grid.state_new.momentum.y .≈ expected_py)
    @test all(grid.state_new.momentum.z .== 0.0)

    # Performance assertions
    @test_call apply_external_forces!(force, grid, dt)
    @test_opt apply_external_forces!(force, grid, dt)
end

@testset "3. Radial Inverse Square Force Field" begin
    reset_grid_mass!(grid, 1.0) # mass = 1.0
    
    # Place the center of attraction exactly at node position [2, 2, 2] 
    # (Which maps to coordinates (1.0, 1.0, 1.0) given origin=0 and dx=1)
    center_coords = SVector{3, T}(1.0, 1.0, 1.0)
    F_0 = -10.0 # Attractive force
    force = RadialInvSquareForceField(F_0, center_coords)

    apply_external_forces!(force, grid, dt)

    # Case A: Center Singularity check
    # Node [2,2,2] is exactly on the center. The force should be 0 due to r_vec = 0.
    @test grid.state_new.momentum.x[2, 2, 2] ≈ 0.0 atol=1e-12
    @test grid.state_new.momentum.y[2, 2, 2] ≈ 0.0 atol=1e-12
    @test grid.state_new.momentum.z[2, 2, 2] ≈ 0.0 atol=1e-12
    # Case B: Directional Attraction check
    # Node [1, 2, 2] is at coordinate (0.0, 1.0, 1.0), which is directly to the LEFT 
    # of the center. An attractive force should pull it to the RIGHT (+x direction).
    @test grid.state_new.momentum.x[1, 2, 2] > 0.0
    @test grid.state_new.momentum.y[1, 2, 2] ≈ 0.0 atol=1e-12
    @test grid.state_new.momentum.z[1, 2, 2] ≈ 0.0 atol=1e-12

    # Node [3, 2, 2] is at coordinate (2.0, 1.0, 1.0), which is directly to the RIGHT 
    # of the center. An attractive force should pull it to the LEFT (-x direction).
    @test grid.state_new.momentum.x[3, 2, 2] < 0.0
    @test grid.state_new.momentum.y[3, 2, 2] ≈ 0.0 atol=1e-12
    @test grid.state_new.momentum.z[3, 2, 2] ≈ 0.0 atol=1e-12
    
    # Performance assertions
    @test_call apply_external_forces!(force, grid, dt)
    @test_opt apply_external_forces!(force, grid, dt)
end
