using SmashMPM
using StaticArrays
using LinearAlgebra
using KernelAbstractions: CPU, Backend
using Base.Threads


# ---------------------------------------------------------------------------- #
#                               Backend Selection                              #
# ---------------------------------------------------------------------------- #
backend_to_use = :cuda
# const backend_to_use = :cuda

T = backend_to_use === :cpu ? Float64 : Float64 # Choose correct precision based on backend

if backend_to_use === :cuda
    println("Using CUDA backend...")
    using CUDA  # Import CUDA only if using CUDABackend
    CUDA.set_runtime_version!(local_toolkit=true)   # true if local CUDA toolkit is installed, false if using system CUDA
    println("CUDA device: ", CUDA.name(CUDA.device()))
    CUDA.allowscalar(false)
    BACKEND = CUDABackend()
else
    println("Using CPU backend using $(Threads.nthreads()) threads...")
    BACKEND = CPU()
end


# ---------------------------------------------------------------------------- #
#                             Simulation Parameters                            #
# ---------------------------------------------------------------------------- #
const DX = T(0.05)
const T_MAX = T(10.0)
const PADDING = 8
const PPC_1D = 2
const cfl_number = T(0.4)
const DT_MAX = T(1e-3)

const SAVE_TIME_INTERVAL = T(0.1)   # Save simulation state every .1 seconds

# ---------------------------------------------------------------------------- #
#                                Cube Parameters                               #
# ---------------------------------------------------------------------------- #
const CUBE_SIZE = T(1.0)
const CUBE_MATERIAL = Basalt(T, eos=:murnaghan, elasticity=:hyperelastic)
# const CUBE_MATERIAL = NeoHookean(E=T(1e6), ν=T(0.3), ρ=T(1000.0))
const CUBE_ROT_SPEED = T(2)  # radians per second


function build_cube(T)
    shape_cube = RectangularPrism{T}(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE)
    body_cube = Body(shape_cube, SVector{3, T}(0.0, 0.0, 0.0), SVector{3, T}(0.0, 0.0, CUBE_ROT_SPEED), CUBE_MATERIAL)
    return body_cube
end

function main(backend=BACKEND, T=T)
    # Create a cube body
    println("Creating cube body...")
    body_cube = build_cube(T)

        exporter = HDF5Exporter(
        output_dir="output", 
        filename_prefix="rotating_cube",
        write_xdmf=true,
        export_grid=true
    )

    # Setup the simulation
    println("Setting up simulation...")
    sim_setup = SimulationSetup(
        dx=DX,
        t_max=T_MAX,
        padding=PADDING,
        ppc_1d=PPC_1D,
        cfl_number=cfl_number,
        dt_max=DT_MAX,
        backend=BACKEND,
        export_time_interval=SAVE_TIME_INTERVAL,
        exporter=exporter
    )

    # Create a simulation with the cube body
    println("Setting up Model...")
    model = build_mpm_model((body_cube,), sim_setup)

    N_particles = length(model.particle_sets[1].particles)
    grid_dimensions = size(model.grid.state_read)
    println("Simulation setup complete. Number of particles: $N_particles, Grid dimensions: $grid_dimensions")


    solve!(model, sim_setup)
    
    println("\nSimulation complete.")
end

main()
