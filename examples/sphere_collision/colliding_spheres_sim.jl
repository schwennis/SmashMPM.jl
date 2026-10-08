using SmashMPM
using StaticArrays
using KernelAbstractions: CPU, Backend


backend_to_use = :cuda
# const backend_to_use = :cpu

T = Float64 # Backend independent precision, has to be Float64 to conserve energy and angular momentum


if backend_to_use === :cuda
    println("Using CUDA backend...")
    using CUDA  # Import CUDA only if using CUDABackend
    CUDA.allowscalar(false)
    BACKEND = CUDABackend()
else
    println("Using CPU backend using $(Threads.nthreads()) threads...")
    BACKEND = CPU()
end





# ---------------------------------------------------------------------------- #
#                               Impact Parameters                              #
# ---------------------------------------------------------------------------- #
const IMPACT_VELOCITY = 1     # m/s
const IMPACT_PARAMETER = 0.5    # Impact parameter (0 = head-on, 1 = grazing)
const SPHERE_RADIUS = 0.5
const INITIAL_SEPARATION = 2.0 * SPHERE_RADIUS + 1.0
time_to_collision = (INITIAL_SEPARATION - 2 * SPHERE_RADIUS) / IMPACT_VELOCITY


# ---------------------------------------------------------------------------- #
#                             Simulation Parameters                            #
# ---------------------------------------------------------------------------- #
const DX = T(0.03)
const T_MAX = T(3*time_to_collision)  # Run simulation for thrice the time to collision
const PADDING = 3
const PPC_1D = 2
const cfl_number = T(0.4)
const DT_MAX = T(1e-3)
const BUFFER_WIDTH = SVector{3, Int}(15, 15, 15)


# ---------------------------------------------------------------------------- #
#                                Export Settings                               #
# ---------------------------------------------------------------------------- #
animation_time = 5 # real seconds
fps = 30
const SAVE_TIME_INTERVAL = T(T_MAX / (animation_time * fps))   # ≈ 3.3e-5 s, 300 Frames


# ---------------------------------------------------------------------------- #
#                               Sphere Materials                               #
# ---------------------------------------------------------------------------- #
const SPHERE1_MATERIAL = Basalt(T, eos=:tillotson, strength=:hypoelastic)
const SPHERE2_MATERIAL = Basalt(T, eos=:tillotson, strength=:hypoelastic)


# ---------------------------------------------------------------------------- #
#                               Sphere Parameters                              #
# ---------------------------------------------------------------------------- #
const SPHERE1_POSITION = SVector{3, T}(-INITIAL_SEPARATION/2, 0.0, 0.0)
const SPHERE1_VELOCITY = SVector{3, T}(IMPACT_VELOCITY/2, 0.0, 0.0)

const SPHERE2_POSITION = SVector{3, T}(INITIAL_SEPARATION/2, IMPACT_PARAMETER * 2 * SPHERE_RADIUS, 0.0)
const SPHERE2_VELOCITY = SVector{3, T}(-IMPACT_VELOCITY/2, 0.0, 0.0)



# ---------------------------------------------------------------------------- #
#                                 Crate Spheres                                #
# ---------------------------------------------------------------------------- #
println("Setting up simulation...")

sphere1_shape = Sphere{T}(SPHERE_RADIUS, SPHERE1_POSITION)
sphere2_shape = Sphere{T}(SPHERE_RADIUS, SPHERE2_POSITION)

sphere1_body = Body(sphere1_shape, SPHERE1_VELOCITY, zero(SVector{3, T}), SPHERE1_MATERIAL)
sphere2_body = Body(sphere2_shape, SPHERE2_VELOCITY, zero(SVector{3, T}), SPHERE2_MATERIAL)

bodies = (sphere1_body, sphere2_body)


exporter = HDF5Exporter(
    output_dir="output",
    filename_prefix = "colliding_spheres",
    write_xdmf = true,
    compression_level = 3,
    export_particles = true,
    export_grid = true
)

boundary_condition = NoSlipBoundary()

sim_setup = SimulationSetup(
    dx=DX,
    padding=PADDING,
    buffer_width=BUFFER_WIDTH,
    cfl_number=cfl_number,
    t_max=T_MAX,
    dt_max=DT_MAX,
    ppc_1d=PPC_1D,
    exporter=exporter,
    export_time_interval=SAVE_TIME_INTERVAL,
    backend=BACKEND,
    boundary_condition=boundary_condition
)

model = build_mpm_model(bodies, sim_setup)

N_particles = length(model.particle_sets[1].particles) + length(model.particle_sets[2].particles)
grid_dimensions = size(model.grid.state_read)
println("Simulation setup complete. Number of particles: $N_particles, Grid dimensions: $grid_dimensions")


solve!(model, sim_setup)