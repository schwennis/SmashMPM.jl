# --- Shared Grid Setup ---
T = Float64
dx = 1.0
N = SVector{3, Int}(4, 4, 4)
origin = SVector{3, T}(0.0, 0.0, 0.0)
padding = 0

grid = DenseGrid(dx, N, origin, padding, CPU())

# Helper function to reset and prime the grid with mass
# External forces are booked into the nodal force (f_I += m_I a), not the momentum.
function reset_grid_mass!(grid, initial_mass=2.0)
    fill!(grid.state_write.mass, initial_mass)
    fill!(grid.state_write.momentum, zero(SVector{3, T}))
    fill!(grid.state_write.force, zero(SVector{3, T}))
end

@testset "1. No External Force" begin
    reset_grid_mass!(grid)
    force = NoExternalForce()

    @test_nowarn apply_external_forces!(force, grid)
    
    # Verify force and momentum remain untouched (completely zero)
    @test all(grid.state_write.force.x .== 0.0)
    @test all(grid.state_write.force.y .== 0.0)
    @test all(grid.state_write.force.z .== 0.0)
    @test all(grid.state_write.momentum .== Ref(zero(SVector{3, T})))

    # Performance assertions
    @test_call apply_external_forces!(force, grid)
    @test_opt apply_external_forces!(force, grid)
end

@testset "2. Constant Gravity" begin
    reset_grid_mass!(grid, 2.0) # mass = 2.0
    g_vec = SVector{3, T}(0.0, -9.81, 0.0)
    force = ConstantGravity(g_vec)

    apply_external_forces!(force, grid)

    # Expected force Δf = m * g = 2.0 * -9.81 = -19.62 (no dt), momentum untouched
    expected_fy = -19.62
    @test all(grid.state_write.force.x .== 0.0)
    @test all(grid.state_write.force.y .≈ expected_fy)
    @test all(grid.state_write.force.z .== 0.0)
    @test all(grid.state_write.momentum .== Ref(zero(SVector{3, T})))

    # Performance assertions
    @test_call apply_external_forces!(force, grid)
    @test_opt apply_external_forces!(force, grid)
end

@testset "3. Radial Inverse Square Force Field" begin
    reset_grid_mass!(grid, 1.0) # mass = 1.0
    
    # Place the center of attraction exactly at node position [2, 2, 2] 
    # (Which maps to coordinates (1.0, 1.0, 1.0) given origin=0 and dx=1)
    center_coords = SVector{3, T}(1.0, 1.0, 1.0)
    F_0 = -10.0 # Attractive force
    force = RadialInvSquareForceField(F_0, center_coords)

    apply_external_forces!(force, grid)

    # Case A: Center Singularity check
    # Node [2,2,2] is exactly on the center. The force should be 0 due to r_vec = 0.
    @test grid.state_write.force.x[2, 2, 2] ≈ 0.0 atol=1e-12
    @test grid.state_write.force.y[2, 2, 2] ≈ 0.0 atol=1e-12
    @test grid.state_write.force.z[2, 2, 2] ≈ 0.0 atol=1e-12
    # Case B: Directional Attraction check
    # Node [1, 2, 2] is at coordinate (0.0, 1.0, 1.0), which is directly to the LEFT 
    # of the center. An attractive force should pull it to the RIGHT (+x direction).
    @test grid.state_write.force.x[1, 2, 2] > 0.0
    @test grid.state_write.force.y[1, 2, 2] ≈ 0.0 atol=1e-12
    @test grid.state_write.force.z[1, 2, 2] ≈ 0.0 atol=1e-12

    # Node [3, 2, 2] is at coordinate (2.0, 1.0, 1.0), which is directly to the RIGHT 
    # of the center. An attractive force should pull it to the LEFT (-x direction).
    @test grid.state_write.force.x[3, 2, 2] < 0.0
    @test grid.state_write.force.y[3, 2, 2] ≈ 0.0 atol=1e-12
    @test grid.state_write.force.z[3, 2, 2] ≈ 0.0 atol=1e-12
    
    # Performance assertions
    @test_nowarn apply_external_forces!(force, grid)
end