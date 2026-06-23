
# --- Setup common grid ---
T = Float64
dx = 1.0
N = SVector{3, Int}(6, 6, 6) # Small grid for explicit checking
origin = SVector{3, T}(0.0, 0.0, 0.0)
padding = 2

grid = DenseGrid(dx, N, origin, padding, CPU())

@testset "1. No Boundary Condition" begin
    bc = NoBoundaryCondition()
    
    # Fill grid with dummy momentum data
    fill!(grid.state_new.momentum, SVector{3, T}(1.0, 1.0, 1.0))
    
    # Applying it should do absolutely nothing
    @test_nowarn apply_boundary_condition!(bc, grid)
    @test all(grid.state_new.momentum .== Ref(SVector{3, T}(1.0, 1.0, 1.0)))
    
    # Performance check
    @test_call apply_boundary_condition!(bc, grid)
    @test_opt apply_boundary_condition!(bc, grid)
end

@testset "2. No-Slip Boundary Mask Construction" begin
    # Typinferred constructor check
    bc = @inferred NoSlipBoundary(grid)
    
    # Verify mask size matches grid size
    @test size(bc.mask) == size(grid.state_new.momentum)
    @test eltype(bc.mask) === Bool

    # Explicitly verify the padding layers are masked (true) 
    # and the inner domain is clear (false)
    @test bc.mask[1, :, :] |> all       # Outer shell boundary
    @test bc.mask[end, :, :] |> all     # Outer shell boundary
    
    # Inner domain check (padding is 2, size is 6, inner is indices 3 and 4)
    @test !any(bc.mask[3:4, 3:4, 3:4]) 
end

@testset "3. No-Slip Application & Correctness" begin
    bc = NoSlipBoundary(grid)
    
    # Fill entire grid momentum with 1.0 vectors
    fill!(grid.state_new.momentum, SVector{3, T}(1.0, 1.0, 1.0))
    
    # Apply BC
    apply_boundary_condition!(grid, bc)
    
    # Verify the padding/ghost region is completely zeroed out
    @test all(grid.state_new.momentum[1, :, :] .== Ref(zero(SVector{3, T})))
    @test all(grid.state_new.momentum[:, 1, :] .== Ref(zero(SVector{3, T})))
    
    # Verify the core active simulation domain remains completely untouched
    @test all(grid.state_new.momentum[3:4, 3:4, 3:4] .== Ref(SVector{3, T}(1.0, 1.0, 1.0)))
end

@testset "4. Performance & Dynamic Dispatch (JET)" begin
    bc = NoSlipBoundary(grid)
    
    # Test type stability of the broadcast call
    @test_call apply_boundary_condition!(grid, bc)
    @test_opt apply_boundary_condition!(grid, bc)
    
    # Quick benchmark verification hook (Should be 0 allocations)
    # using BenchmarkTools
    # @test (@allocated apply_boundary_condition!(grid, bc)) == 0
end
