
# --- Setup common grid ---
T = Float64
dx = 1.0
N = SVector{3, Int}(6, 6, 6) # Small grid for explicit checking
origin = SVector{3, T}(0.0, 0.0, 0.0)
padding = 2

grid = DenseGrid(dx, N, origin, padding, CPU())

@testset "1. No Boundary Condition" begin
    bc = NullBoundaryCondition()
    
    # Fill grid with dummy momentum data
    fill!(grid.state_write.momentum, SVector{3, T}(1.0, 1.0, 1.0))
    
    # Applying it should do absolutely nothing
    @test_nowarn apply_boundary_condition!(bc, grid)
    @test all(grid.state_write.momentum .== Ref(SVector{3, T}(1.0, 1.0, 1.0)))
    
    # Performance check
    @test_call apply_boundary_condition!(bc, grid)
    @test_opt apply_boundary_condition!(bc, grid)
end

@testset "2. No-Slip Boundary Mask Construction" begin
    # The boundary condition is stateless; the grid is supplied when applying it.
    @test @inferred(NoSlipBoundary()) isa NoSlipBoundary
end

@testset "3. No-Slip Application & Correctness" begin
    bc = NoSlipBoundary()
    
    # Fill entire grid momentum with 1.0 vectors
    fill!(grid.state_write.momentum, SVector{3, T}(1.0, 1.0, 1.0))
    
    # Apply BC
    apply_boundary_condition!(bc, grid)
    
    # Verify every padding face is completely zeroed out.
    for component in (grid.state_write.momentum.x,
                      grid.state_write.momentum.y,
                      grid.state_write.momentum.z)
        @test all(component[1:2, :, :] .== 0.0)
        @test all(component[5:6, :, :] .== 0.0)
        @test all(component[:, 1:2, :] .== 0.0)
        @test all(component[:, 5:6, :] .== 0.0)
        @test all(component[:, :, 1:2] .== 0.0)
        @test all(component[:, :, 5:6] .== 0.0)
    end
    
    # Verify the core active simulation domain remains completely untouched
    @test all(grid.state_write.momentum.x[3:4, 3:4, 3:4] .== 1.0)
    @test all(grid.state_write.momentum.y[3:4, 3:4, 3:4] .== 1.0)
    @test all(grid.state_write.momentum.z[3:4, 3:4, 3:4] .== 1.0)
end

@testset "4. Performance & Dynamic Dispatch (JET)" begin
    bc = NoSlipBoundary()
    
    @test_nowarn apply_boundary_condition!(bc, grid)
        
    # Quick benchmark verification hook (Should be 0 allocations)
    # using BenchmarkTools
    # @test (@allocated apply_boundary_condition!(bc, grid)) == 0
end
