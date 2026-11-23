
# --- Shared Grid Setup ---
T = Float64
dx = 1.0
N = SVector{3, Int}(4, 4, 4)
origin = SVector{3, T}(0.0, 0.0, 0.0)
padding = 0
dt = 0.1

grid = DenseGrid(dx, N, origin, padding, CPU())

# Helper function to reset and prime the grid with mass.
# External forces write the nodal force f_I = m_I * a(x_I) into state_write.force
# (no dt); the momentum is only changed later by grid_kick!.
function reset_grid_mass!(grid, initial_mass=2.0)
    fill!(grid.state_write.mass, initial_mass)
    fill!(grid.state_write.momentum, zero(SVector{3, T}))
    fill!(grid.state_write.force, zero(SVector{3, T}))
end

momentum_untouched(grid) = all(grid.state_write.momentum .== Ref(zero(SVector{3, T})))

@testset "1. No External Force" begin
    reset_grid_mass!(grid)
    force = NoExternalForce()

    @test_nowarn apply_external_forces!(force, grid, dt)
    
    # Verify force and momentum remain untouched (completely zero)
    @test all(grid.state_write.force .== Ref(zero(SVector{3, T})))
    @test momentum_untouched(grid)

    # Performance assertions
    @test_call apply_external_forces!(force, grid, dt)
    @test_opt apply_external_forces!(force, grid, dt)
end

@testset "2. Constant Gravity" begin
    reset_grid_mass!(grid, 2.0) # mass = 2.0
    g_vec = SVector{3, T}(0.0, -9.81, 0.0)
    force = ConstantGravity(g_vec)

    apply_external_forces!(force, grid, dt)

    # Expected nodal force f = m * g = 2.0 * -9.81 = -19.62 (independent of dt)
    expected_fy = -19.62
    @test all(grid.state_write.force.x .== 0.0)
    @test all(grid.state_write.force.y .≈ expected_fy)
    @test all(grid.state_write.force.z .== 0.0)
    @test momentum_untouched(grid)

    # Forces accumulate (internal force from g2p2g + external force)
    apply_external_forces!(force, grid, dt)
    @test all(grid.state_write.force.y .≈ 2 * expected_fy)

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
    f = grid.state_write.force

    # Case A: Center Singularity check
    # Node [2,2,2] is exactly on the center. The force should be 0 due to r_vec = 0.
    @test f.x[2, 2, 2] ≈ 0.0 atol=1e-12
    @test f.y[2, 2, 2] ≈ 0.0 atol=1e-12
    @test f.z[2, 2, 2] ≈ 0.0 atol=1e-12
    # Case B: Directional Attraction check
    # Node [1, 2, 2] is at coordinate (0.0, 1.0, 1.0), which is directly to the LEFT 
    # of the center. An attractive force should pull it to the RIGHT (+x direction).
    @test f.x[1, 2, 2] > 0.0
    @test f.y[1, 2, 2] ≈ 0.0 atol=1e-12
    @test f.z[1, 2, 2] ≈ 0.0 atol=1e-12

    # Node [3, 2, 2] is at coordinate (2.0, 1.0, 1.0), which is directly to the RIGHT 
    # of the center. An attractive force should pull it to the LEFT (-x direction).
    @test f.x[3, 2, 2] < 0.0
    @test f.y[3, 2, 2] ≈ 0.0 atol=1e-12
    @test f.z[3, 2, 2] ≈ 0.0 atol=1e-12

    # Symmetry: both neighbours at distance 1 get forces of equal magnitude
    @test f.x[1, 2, 2] ≈ -f.x[3, 2, 2]
    @test momentum_untouched(grid)
    
    # Performance assertions
    @test_nowarn apply_external_forces!(force, grid, dt)
end

@testset "4. Vector Field Force" begin
    reset_grid_mass!(grid, 2.0)
    a = SVector{3, T}(1.0, -2.0, 0.5)
    force = VectorFieldForce(fill(a, size(grid.state_write.mass)))

    apply_external_forces!(force, grid, dt)

    # f = m * a, written to the force field and not to the momentum
    @test all(grid.state_write.force .≈ Ref(2.0 * a))
    @test momentum_untouched(grid)

    # Nodes without mass get no force
    reset_grid_mass!(grid, 0.0)
    apply_external_forces!(force, grid, dt)
    @test all(grid.state_write.force .== Ref(zero(SVector{3, T})))
end
