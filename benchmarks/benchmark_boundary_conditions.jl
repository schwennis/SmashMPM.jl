using BenchmarkTools
using InteractiveUtils
using Profile
using PProf
using StaticArrays

include("../src/SmashMPM.jl")
using .SmashMPM


dx = 1.0
N = @SVector [128, 128, 128]
origin = @SVector [0.0, 0.0, 0.0]
padding = 4


grid = DenseGrid(dx, N, origin, padding)

# fill the grid with some dummy data for benchmarking
function populate_dummy_data!(grid, N)
    # Inside a function, variables are local and perfectly type-inferred
    for i in 1:N[1], j in 1:N[2], k in 1:N[3]
        grid.state_old.mass[i, j, k] = rand()
        grid.state_old.wave_speed[i, j, k] = rand() * 10
    end
end

populate_dummy_data!(grid, N)

bc = NoSlipBoundary(grid)

println("Benchmarking apply_boundary_condition!...")
# Warm-up call to compile the function
apply_boundary_condition!(grid, bc)
display(@benchmark apply_boundary_condition!($grid, $bc))

Profile.clear_malloc_data()
apply_boundary_condition!(grid, bc)

