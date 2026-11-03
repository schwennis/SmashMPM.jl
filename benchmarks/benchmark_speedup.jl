# File: benchmarks/benchmark_speedup.jl
#
# Scaling benchmark: times the G2P2G kernel and a full time step for growing
# particle counts on every backend and prints the speedup of each accelerator
# relative to the CPU. Replaces the old benchmark_gpu.jl.

# ---------------------------------------------------------------------------- #
#                                   Results                                    #
# ---------------------------------------------------------------------------- #
"""
    SpeedupRow

One problem size of the speedup benchmark. The times are median times in ns per
call, stored per backend name.
"""
struct SpeedupRow
    n_particles::Int
    grid::String
    grid_size::String
    g2p2g::Dict{String, Float64}
    step::Dict{String, Float64}
end

# Grid configurations of the speedup benchmark (open domain, no walls)
const SPEEDUP_GRIDS = (
    ("DenseGrid",  DenseGrid),
    ("SparseGrid", SparseGrid),
)

grid_size_label(grid::DenseGrid)  = join(size(grid.state_read.mass), "×")
grid_size_label(grid::SparseGrid) = "$(grid.n_active_read) tiles"


# ---------------------------------------------------------------------------- #
#                                    Timing                                    #
# ---------------------------------------------------------------------------- #
"""
    time_model(model; seconds=2.0) -> (t_g2p2g, t_step)

Median times (ns) of `g2p2g!` and `step!`. The particles are restored before
every sample, so all samples start from the same state.
"""
function time_model(model::MPMModel; seconds::Real=2.0)
    backup  = capture_particle_backup(model)
    dt_tiny = typeof(model.t)(1e-7)

    # warm up: compile all kernels before measuring
    restore_particles!(model, backup); g2p2g!(model, dt_tiny)
    restore_particles!(model, backup); step!(model)

    b_g2p2g = @benchmarkable(g2p2g!($model, $dt_tiny), setup = (restore_particles!($model, $backup)))
    b_step  = @benchmarkable(step!($model),            setup = (restore_particles!($model, $backup)))

    t_g2p2g = median(run(b_g2p2g; seconds=seconds, evals=1)).time
    t_step  = median(run(b_step;  seconds=seconds, evals=1)).time
    restore_particles!(model, backup)
    return t_g2p2g, t_step
end


# ---------------------------------------------------------------------------- #
#                                    Driver                                    #
# ---------------------------------------------------------------------------- #
"""
    run_speedup_benchmarks(backends; T=Float32, particle_counts=[...], material=nothing,
                           seconds=2.0, io=stdout) -> Vector{SpeedupRow}

Run the scaling benchmark on all `backends` (the first one is the reference,
normally the CPU) and print one speedup table per accelerator. `material`
defaults to Basalt with Tillotson EoS, hypoelasticity and bulk viscosity.
"""
function run_speedup_benchmarks(
    backends;
    T::Type{<:AbstractFloat}=Float32,
    particle_counts::Vector{Int}=[10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
    material=nothing,
    seconds::Real=2.0,
    io::IO=stdout,
)
    if length(backends) < 2
        @warn "Speedup benchmark needs a GPU backend in addition to the CPU, skipping it."
        return SpeedupRow[]
    end

    mat = isnothing(material) ?
        Basalt(T; eos=:tillotson, elasticity=:hypoelastic, viscosity=BulkViscosity{T}()) : material
    gravity = ConstantGravity(SVector{3, T}(0.0, -9.81, 0.0))

    rows = SpeedupRow[]
    for n_target in particle_counts, (grid_label, grid_type) in SPEEDUP_GRIDS
        g2p2g_times = Dict{String, Float64}()
        step_times  = Dict{String, Float64}()
        n_p, size_label = 0, ""

        for (b_name, backend) in backends
            @info "Speedup benchmark: $(n_target) particles, $(grid_label), $(b_name)"
            model, _ = build_benchmark_model(n_target, backend; T=T, grid_type=grid_type,
                                             material=mat, external_force=gravity)
            n_p        = n_particles(model)
            size_label = grid_size_label(model.grid)

            g2p2g_times[b_name], step_times[b_name] = time_model(model; seconds=seconds)

            model = nothing
            reclaim_device_memory!(backend)
        end
        push!(rows, SpeedupRow(n_p, grid_label, size_label, g2p2g_times, step_times))
    end

    reference = first(backends)[1]
    for (b_name, _) in backends[2:end]
        print_speedup_table(rows, reference, b_name; io=io)
    end
    return rows
end


# ---------------------------------------------------------------------------- #
#                                    Output                                    #
# ---------------------------------------------------------------------------- #
function print_speedup_table(rows::Vector{SpeedupRow}, reference::String, accelerator::String; io::IO=stdout)
    width = 132
    println(io, "\n", "="^width)
    println(io, " $(reference) vs. $(accelerator) SPEEDUP (median time per call, CPU threads: $(Threads.nthreads()))")
    println(io, "-"^width)
    @printf(io, "%-10s | %-10s | %-14s | %-12s | %-12s | %-9s | %-12s | %-12s | %-9s\n",
            "Particles", "Grid", "Size",
            "g2p2g! $(reference)", "g2p2g! $(accelerator)", "Speedup",
            "step! $(reference)",  "step! $(accelerator)",  "Speedup")
    println(io, "-"^width)
    for row in rows
        @printf(io, "%-10d | %-10s | %-14s | %-12s | %-12s | %-9s | %-12s | %-12s | %-9s\n",
                row.n_particles, row.grid, row.grid_size,
                format_time(row.g2p2g[reference]), format_time(row.g2p2g[accelerator]),
                format_speedup(row.g2p2g[reference] / row.g2p2g[accelerator]),
                format_time(row.step[reference]), format_time(row.step[accelerator]),
                format_speedup(row.step[reference] / row.step[accelerator]))
    end
    println(io, "="^width)
    return nothing
end
