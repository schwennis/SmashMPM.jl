using SmashMPM
using StaticArrays
using KernelAbstractions: CPU, Backend


backend_to_use = :cpu
# const backend_to_use = :cuda

T = Float64 # Backend independent precision


if backend_to_use === :cuda
    println("Using CUDA backend...")
    using CUDA  # Import CUDA only if using CUDABackend
    CUDA.set_runtime_version!(local_toolkit=true)   # true if local CUDA toolkit is installed, false if using system CUDA
    CUDA.allowscalar(false)
    BACKEND = CUDABackend()
else
    println("Using CPU backend using $(Threads.nthreads()) threads...")
    BACKEND = CPU()
end


# ---------------------------------------------------------------------------- #
#                               Tank Parameters                                #
# ---------------------------------------------------------------------------- #
# Axes: x = length, y = width, z = height (gravity points in -z)
const GRAVITY = T(9.81)                 # m/s^2

const TANK_SIZE_X = T(10.0)             # length of the tank in m
const TANK_SIZE_Y = T(5.0)              # width of the tank in m
const TANK_SIZE_Z = T(5.0)              # height of the tank in m

const WATER_SIZE_X = T(3.0)             # m, initial length of the water column
const WATER_SIZE_Y = TANK_SIZE_Y        # m, initial width of the water column (wall to wall)
const WATER_SIZE_Z = T(0.7) * TANK_SIZE_Z   # m, initial height of the water column

# Column with a square cross-section (x, y) over the full height of the tank, clamped at the floor and at the top
const COLUMN_WIDTH = T(1.0)             # m, edge length of the cross-section
const CLAMPED_CELLS = 2                 # Number of cells the column reaches into the walls (floor and top).

const INITIAL_SEPARATION = T(2.0)       # m, distance between water column and column


# ---------------------------------------------------------------------------- #
#                            Simulation Parameters                             #
# ---------------------------------------------------------------------------- #
const DX = T(0.08)                       # m, DX = 0.02 would be ≈ 25 million water particles
const PPC_1D = 2
const PADDING = CLAMPED_CELLS + 2
const CFL_NUMBER = T(0.4)
const DT_MAX = T(1e-3)
const T_MAX = T(4.0)                   # s, the front hits the column after ≈ 0.5 s

const N_EXPORTS = T_MAX * T(30)
const EXPORT_GRID = false


# ---------------------------------------------------------------------------- #
#                                  Materials                                   #
# ---------------------------------------------------------------------------- #
# Artificial sound speed: ≈ 10 times the free fall velocity of the water column
const WATER_SOUNDSPEED = 10 * sqrt(2 * GRAVITY * WATER_SIZE_Z)  # m/s
const WATER_DENSITY = T(1000.0)         # kg/m^3

# Weakly compressible water without shear strength (not exported by SmashMPM, hence the prefix)
const WATER_EOS = MurnaghanEOS(WATER_DENSITY, WATER_DENSITY * WATER_SOUNDSPEED^2, T(7), T(1))
const WATER_ELASTICITY = HypoElasticity(zero(T))
const WATER_MATERIAL = SolidMaterial(; eos=WATER_EOS, elasticity=WATER_ELASTICITY,
                                     plasticity=SmashMPM.NullPlasticity(), damage=SmashMPM.NoDamage(),
                                     viscosity=BulkViscosity(), ρ=WATER_DENSITY)

# Stiff column, the water pressure is of the order of 1e4 Pa (E = 2e7 Pa: deflection of a few cm)
const COLUMN_MATERIAL = NeoHookean(E=T(2e7), ν=T(0.3), ρ=T(2500.0))


# ---------------------------------------------------------------------------- #
#                                    Checks                                    #
# ---------------------------------------------------------------------------- #
const COLUMN_X_MAX = WATER_SIZE_X + INITIAL_SEPARATION + COLUMN_WIDTH

@assert COLUMN_X_MAX < TANK_SIZE_X "Water column, separation and column must fit into TANK_SIZE_X."
@assert WATER_SIZE_Z <= TANK_SIZE_Z "WATER_SIZE_Z must not exceed TANK_SIZE_Z."
@assert COLUMN_WIDTH <= TANK_SIZE_Y "The column must fit into the tank."
@assert COLUMN_WIDTH >= 2 * DX "COLUMN_WIDTH must be at least 2 * DX to avoid numerical issues."
@assert INITIAL_SEPARATION >= 2 * DX "INITIAL_SEPARATION must be at least 2 * DX to avoid numerical issues."
@assert 0 < CLAMPED_CELLS < PADDING "CLAMPED_CELLS must be positive and smaller than PADDING."


# ---------------------------------------------------------------------------- #
#                                    Bodies                                    #
# ---------------------------------------------------------------------------- #
println("Creating bodies...")

# Water column in the corner x = y = z = 0 of the tank, spanning the full width
water_center = SVector{3, T}(WATER_SIZE_X / 2, WATER_SIZE_Y / 2, WATER_SIZE_Z / 2)
water_shape = RectangularPrism(WATER_SIZE_X, WATER_SIZE_Y, WATER_SIZE_Z, water_center, zero(SVector{3, T}))
water_body = Body(water_shape, zero(SVector{3, T}), zero(SVector{3, T}), WATER_MATERIAL)

# Column, centered in y. It reaches CLAMPED_CELLS cells into the floor and into the top wall of the
# grid (padding nodes have zero momentum), which clamps both ends.
column_height = TANK_SIZE_Z + 2 * CLAMPED_CELLS * DX
column_center = SVector{3, T}(WATER_SIZE_X + INITIAL_SEPARATION + COLUMN_WIDTH / 2, TANK_SIZE_Y / 2, TANK_SIZE_Z / 2)
column_shape = RectangularPrism(COLUMN_WIDTH, COLUMN_WIDTH, column_height, column_center, zero(SVector{3, T}))
column_body = Body(column_shape, zero(SVector{3, T}), zero(SVector{3, T}), COLUMN_MATERIAL)

bodies = (water_body, column_body)


# ---------------------------------------------------------------------------- #
#                                    Domain                                    #
# ---------------------------------------------------------------------------- #
# The grid is fitted around the bodies (FittedBox), the walls lie `PADDING` nodes behind the outermost
# particles plus the buffer.
#   x: no buffer at the lower side, so the water touches the wall at x = 0. Behind the column the
#      empty space is added via BUFFER_WIDTH_HI.
#   y: the water spans the full width and touches both walls.
#   z: the column reaches CLAMPED_CELLS cells beyond floor and top. The negative buffer moves the
#      wall back to the end of the column, so the floor is at z = 0 and the top wall at z = TANK_SIZE_Z.
const BUFFER_WIDTH = SVector{3, Int}(0, 0, -CLAMPED_CELLS)
const BUFFER_WIDTH_HI = SVector{3, Int}(
    round(Int, (TANK_SIZE_X - COLUMN_X_MAX) / DX),
    0,
    -CLAMPED_CELLS,
)


# ---------------------------------------------------------------------------- #
#                                   Exporter                                   #
# ---------------------------------------------------------------------------- #
exporter = HDF5Exporter(
    output_dir=joinpath(@__DIR__, "output"),
    filename_prefix="dam_break",
    write_xdmf=true,
    compression_level=5,
    export_grid=EXPORT_GRID
)


println("Setting up simulation...")
sim_setup = SimulationSetup(
    dx                      = DX,
    grid_type               = SparseGrid,
    padding                 = PADDING,
    buffer_width            = BUFFER_WIDTH,
    buffer_width_hi         = BUFFER_WIDTH_HI,
    domain                  = FittedBox(),
    cfl_number              = CFL_NUMBER,
    t_max                   = T_MAX,
    dt_max                  = DT_MAX,
    ppc_1d                  = PPC_1D,
    exporter                = exporter,
    export_time_interval    = T(T_MAX / N_EXPORTS),
    boundary_condition      = NoSlipBoundary(),
    external_force          = ConstantGravity(SVector{3, T}(0.0, 0.0, -GRAVITY)),
    backend                 = BACKEND,
)

model = build_mpm_model(bodies, sim_setup)

N_water = length(model.particle_sets[1].particles)
N_total = sum(length(ps.particles) for ps in model.particle_sets)

println("Simulation set up with $(length(bodies)) bodies.")
println("Water particles: $N_water, total number of particles: $N_total\nGrid dimensions: $(size(model.grid.state_read))")
println("Simulation time: $T_MAX s, Time step limit: $DT_MAX s, CFL number: $CFL_NUMBER, water sound speed: $(round(WATER_SOUNDSPEED, digits=1)) m/s")


# ---------------------------------------------------------------------------- #
#                                     Run                                      #
# ---------------------------------------------------------------------------- #
solve!(model, sim_setup)


println("Simulation complete. Exported data to $(exporter.output_dir).")