using BenchmarkTools
using InteractiveUtils
using Profile
using PProf
using StaticArrays

include("../src/SmashMPM.jl")
using .SmashMPM

dx = 1.0
N = @SVector [256, 256, 256]
origin = @SVector [0.0, 0.0, 0.0]
padding = 0


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

println("Benchmarking max_wavespeed...")
max_speed = SmashMPM.max_wavespeed(grid) # Warm-up call to compile the function
display(@benchmark SmashMPM.max_wavespeed($grid))


println("Benchmarking grid_reset!...")
# Warm-up call to compile the function
SmashMPM.grid_reset!(grid)
display(@benchmark SmashMPM.grid_reset!($grid))