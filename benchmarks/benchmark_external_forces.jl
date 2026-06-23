using BenchmarkTools
using InteractiveUtils
using Profile
using PProf
using StaticArrays

include("../src/SmashMPM.jl")
using .SmashMPM



# --- Setup Benchmarking Harness ---
function run_force_benchmarks(N_size)
    T = Float64
    dx = 1.0
    N = SVector{3, Int}(N_size, N_size, N_size)
    origin = SVector{3, T}(0.0, 0.0, 0.0)
    padding = 0
    dt = 1e-3

    grid = DenseGrid(dx, N, origin, padding, Array)
    
    # Pre-fill with local scope to avoid global allocations
    fill!(grid.state_new.mass, 2.0)
    fill!(grid.state_new.momentum, zero(SVector{3, T}))

    # Instantiate Force structures
    no_force = NoExternalForce()
    gravity  = ConstantGravity(SVector{3, T}(0.0, -9.81, 0.0))
    radial   = RadialInvSquareForceField(-10.0, SVector{3, T}(N_size/2, N_size/2, N_size/2))

    println("="^50)
    println(" BENCHMARKING GRID SIZE: $(N_size)³ ($(N_size^3) nodes)")
    println("="^50)

    println("\n--> Testing NoExternalForce:")
    # Warmup
    apply_external_forces!(no_force, grid, dt)
    @btime apply_external_forces!($no_force, $grid, $dt)

    println("\n--> Testing ConstantGravity:")
    # Warmup
    apply_external_forces!(gravity, grid, dt)
    @btime apply_external_forces!($gravity, $grid, $dt)

    println("\n--> Testing RadialInvSquareForceField:")
    # Warmup
    apply_external_forces!(radial, grid, dt)
    @btime apply_external_forces!($radial, $grid, $dt)
end

# --- Execution ---

# 1. Cache-Friendly Size (32,768 nodes)
run_force_benchmarks(32)

# 2. Production Scale Size (2,097,152 nodes)
run_force_benchmarks(128)