# File: benchmarks/bench_utils.jl
using BenchmarkTools
using KernelAbstractions
using StaticArrays
using StructArrays
using LinearAlgebra
using Printf
using SmashMPM

# ---------------------------------------------------------------------------- #
#                              Backend Detection                               #
# ---------------------------------------------------------------------------- #
"""
    detect_backends(; cpu_only=false) -> Vector{Tuple{String, Backend}}

Returns the backends to benchmark. The CPU is always included, CUDA only if
CUDA.jl can be loaded and a functional device is found.
"""
function detect_backends(; cpu_only::Bool=false)
    backends = Tuple{String, Backend}[("CPU", CPU())]
    cpu_only && return backends

    cuda_loaded = try
        @eval Main using CUDA
        true
    catch
        false
    end

    # CUDA is loaded at runtime, so it has to be accessed in the latest world age
    cuda_backend = cuda_loaded ? Base.invokelatest(_cuda_backend) : nothing
    if isnothing(cuda_backend)
        @info "CUDA not available, benchmarks run on the CPU only ($(Threads.nthreads()) threads)."
    else
        push!(backends, ("CUDA", cuda_backend))
    end
    return backends
end

function _cuda_backend()
    cuda = getglobal(Main, :CUDA)
    cuda.functional() || return nothing
    cuda.allowscalar(false)
    @info "CUDA available: $(cuda.name(cuda.device())) (added to the benchmark suite)"
    return cuda.CUDABackend()
end

_cuda_reclaim() = (getglobal(Main, :CUDA).reclaim(); nothing)

is_cpu(backend::Backend) = backend isa KernelAbstractions.CPU

"""
    sync_run!(f, backend, args...)

Calls `f(args...)` and synchronizes the backend afterwards. Needed for operations
like `fill!`, `maximum` or broadcasts on the GPU, which do not synchronize themselves.
"""
@inline function sync_run!(f::F, backend::Backend, args...) where {F}
    res = f(args...)
    KernelAbstractions.synchronize(backend)
    return res
end

"""Free device memory between large benchmark groups."""
function reclaim_device_memory!(backend::Backend)
    GC.gc()
    nameof(typeof(backend)) === :CUDABackend && Base.invokelatest(_cuda_reclaim)
    return nothing
end


# ---------------------------------------------------------------------------- #
#                           Allocation Expectations                            #
# ---------------------------------------------------------------------------- #
# Benchmarks that must not allocate are marked with a tag on their parent group:
#   expect_no_allocs!(grp, "grid_reset!")
# The tags survive `run`, `save` and `load`, so the result tables can color the
# allocation column:
#   green       0 allocations, as required
#   red         allocations that should not have happened
#   uncolored   allocations are not checked (kernel launches, GPU calls, ...)
const NO_ALLOC_TAG = "no_allocs"

no_alloc_tag(key) = "$(NO_ALLOC_TAG):$(key)"

"""
    expect_no_allocs!(grp, keys...)

Mark the benchmarks `keys` of `grp` as allocation-free. Without keys, all
benchmarks currently in `grp` (not in its subgroups) are marked.
"""
function expect_no_allocs!(grp::BenchmarkGroup, keys...)
    keys = isempty(keys) ? [k for (k, v) in grp if !(v isa BenchmarkGroup)] : collect(keys)
    for key in keys
        haskey(grp, key) || error("expect_no_allocs!: no benchmark \"$key\" in this group")
        push!(grp.tags, no_alloc_tag(key))
    end
    return grp
end

expects_no_allocs(grp::BenchmarkGroup, key) = no_alloc_tag(key) in grp.tags


# ---------------------------------------------------------------------------- #
#                                Grid Factories                                #
# ---------------------------------------------------------------------------- #
"""
    populate_dense_grid!(grid, backend)

Fill both states of a `DenseGrid` with constant mass and momentum.
"""
function populate_dense_grid!(grid::DenseGrid{T}, backend::Backend) where {T}
    dims = size(grid.state_read.mass)
    for state in (grid.state_read, grid.state_write)
        copyto!(state.mass,       fill(T(2.5),   dims))
        copyto!(state.momentum.x, fill(T(1.0),   dims))
        copyto!(state.momentum.y, fill(T(-2.0),  dims))
        copyto!(state.momentum.z, fill(T(0.5),   dims))
        copyto!(state.force.x,    fill(T(0.0),   dims))
        copyto!(state.force.y,    fill(T(0.0),   dims))
        copyto!(state.force.z,    fill(T(0.0),   dims))
    end
    KernelAbstractions.synchronize(backend)
    return grid
end

"""Fill the active tiles of the write state of a `SparseGrid` with constant values."""
function _populate_active_write!(grid::SparseGrid{T}) where {T}
    used  = 1:grid.n_active_write
    state = grid.state_write
    fill!(view(state.mass,       :, used), T(2.5))
    fill!(view(state.momentum.x, :, used), T(1.0))
    fill!(view(state.momentum.y, :, used), T(-2.0))
    fill!(view(state.momentum.z, :, used), T(0.5))
    fill!(view(state.force.x,    :, used), T(0.0))
    fill!(view(state.force.y,    :, used), T(0.0))
    fill!(view(state.force.z,    :, used), T(0.0))
    return grid
end

"""
    make_populated_sparse_grid(n_target, backend; T=Float32, domain_type=:box, padding=2)
        -> (grid, particle_sets, spline, N)

Build a `SparseGrid` around a sphere of about `n_target` particles. The tiles of
both states are activated and filled with constant values. `domain_type` is
`:box` (BoxDomain of N nodes) or `:open` (OpenDomain).
"""
function make_populated_sparse_grid(
    n_target::Int,
    backend::Backend;
    T::Type{<:AbstractFloat}=Float32,
    domain_type::Symbol=:box,
    padding::Int=2,
)
    domain_type in (:box, :open) || error("Unknown domain_type :$domain_type, use :box or :open")

    R       = T(0.5)
    ppc_1d  = 2
    spacing = T((T(4 / 3 * π) * R^3 / n_target)^(1 / 3))
    dx      = spacing * ppc_1d

    shape = Sphere{T}(radius=R, center=zero(SVector{3, T}))
    pos_cpu, _, _, _ = generate_particles(shape, spacing, T(1000.0), zero(SVector{3, T}))

    # Same grid geometry as in build_mpm_model
    min_c, max_c = SmashMPM.bounding_box(pos_cpu)
    N      = SVector{3, Int}(ceil.(Int, (max_c - min_c .+ spacing) ./ dx)) .+ 2 * padding
    origin = min_c .- padding * dx .- T(0.5) * spacing

    domain    = domain_type === :box ? BoxDomain(N) : OpenDomain()
    spline    = QuadraticSpline()
    n_tiles   = SmashMPM.count_tiles(pos_cpu, dx, origin, spline; domain)
    max_tiles = max(64, cld(5 * n_tiles, 4))        # 25 % headroom, as in build_mpm_model
    grid      = SparseGrid(dx, origin; domain, padding, backend, max_tiles)

    # Minimal particle set, activate_tiles! only reads p_set.particles.pos
    pos_sa = StructArray{SVector{3, T}}((
        x = SmashMPM._to_backend(backend, getindex.(pos_cpu, 1)),
        y = SmashMPM._to_backend(backend, getindex.(pos_cpu, 2)),
        z = SmashMPM._to_backend(backend, getindex.(pos_cpu, 3)),
    ))
    particle_sets = ((particles=(pos=pos_sa,),),)

    # Populate the write state, swap it to read and populate the new write state
    activate_tiles!(grid, particle_sets, spline)
    _populate_active_write!(grid)
    grid_reset!(grid, particle_sets, spline)
    _populate_active_write!(grid)

    KernelAbstractions.synchronize(backend)
    return grid, particle_sets, spline, N
end

"""Clear the tile activation of the write state (setup for `activate_tiles!` benchmarks)."""
function reset_tile_activation!(grid::SparseGrid, backend::Backend)
    SmashMPM.clear!(grid.map_write)
    grid.n_active_write = 0
    KernelAbstractions.synchronize(backend)
    return nothing
end


# ---------------------------------------------------------------------------- #
#                               Model Factories                                #
# ---------------------------------------------------------------------------- #
"""
    build_benchmark_model(n_target, backend; T=Float32, grid_type=DenseGrid, domain=AutoDomain(),
                          material=NeoHookean(...), boundary_condition=NullBoundaryCondition(),
                          external_force=NoExternalForce()) -> (model, setup)

Build a complete `MPMModel` of a rotating sphere with about `n_target` particles.
"""
function build_benchmark_model(
    n_target::Int,
    backend::Backend;
    T::Type{<:AbstractFloat}=Float32,
    grid_type::Type{<:AbstractGrid}=DenseGrid,
    domain=AutoDomain(),
    material=NeoHookean(ρ=T(1000.0), E=T(1e6), ν=T(0.3)),
    boundary_condition=NullBoundaryCondition(),
    external_force=NoExternalForce(),
)
    R       = T(0.5)
    ppc_1d  = 2
    spacing = T((T(4 / 3 * π) * R^3 / n_target)^(1 / 3))
    dx      = spacing * ppc_1d

    shape = Sphere{T}(radius=R, center=zero(SVector{3, T}))
    v0    = SVector{3, T}(0.5, -0.5, 0.0)
    omega = SVector{3, T}(0.0, 0.0, 1.0)
    body  = Body(shape, v0, omega, material)

    setup = SimulationSetup(
        dx=dx,
        grid_type=grid_type,
        domain=domain,
        t_max=T(1e6),                           # prevents dt -> 0 after many benchmark iterations
        padding=3,
        buffer_width=SVector{3, Int}(4, 4, 4),
        ppc_1d=ppc_1d,
        cfl_number=T(0.4),
        dt_max=T(1e-5),
        boundary_condition=boundary_condition,
        external_force=external_force,
        backend=backend,
    )

    model = build_mpm_model((body,), setup)
    KernelAbstractions.synchronize(backend)
    return model, setup
end

"""Number of particles in all particle sets of `model`."""
n_particles(model::MPMModel) = sum(ps -> length(ps.particles.mass), model.particle_sets)


# ---------------------------------------------------------------------------- #
#                          Particle Backup / Restore                           #
# ---------------------------------------------------------------------------- #
# Solver benchmarks change the particles. They are restored in the `setup` of
# every sample, so all samples start from the same state.

"""Copy positions, deformation gradients, material states and wave speeds of the first particle set."""
function capture_particle_backup(model::MPMModel)
    pset = model.particle_sets[1].particles
    return (
        x          = copy(pset.pos.x),
        y          = copy(pset.pos.y),
        z          = copy(pset.pos.z),
        F          = copy(pset.F),
        mat_state  = copy(pset.mat_state),
        wave_speed = copy(pset.wave_speed),     # step! takes its dt from these
    )
end

"""
    restore_particles!(model, backup)

Restore the particles from `backup`, reset the time and clear the write state of
the grid. The read state and the active tiles are kept, they still match the
restored positions since benchmarks only use tiny time steps.
"""
function restore_particles!(model::MPMModel, backup)
    pset = model.particle_sets[1].particles
    copyto!(pset.pos.x, backup.x)
    copyto!(pset.pos.y, backup.y)
    copyto!(pset.pos.z, backup.z)
    copyto!(pset.F, backup.F)
    copyto!(pset.mat_state, backup.mat_state)
    copyto!(pset.wave_speed, backup.wave_speed)
    model.t = zero(model.t)
    clear_write_state!(model.grid)
    KernelAbstractions.synchronize(model.backend)
    return nothing
end

"""Zero the write state of the grid (for the SparseGrid only the active tiles)."""
function clear_write_state!(grid::DenseGrid)
    state = grid.state_write
    for field in (state.mass, state.momentum.x, state.momentum.y, state.momentum.z, state.force.x, state.force.y, state.force.z)
        fill!(field, zero(eltype(field)))
    end
    return nothing
end

function clear_write_state!(grid::SparseGrid)
    state = grid.state_write
    used  = 1:grid.n_active_write
    for field in (state.mass, state.momentum.x, state.momentum.y, state.momentum.z, state.force.x, state.force.y, state.force.z)
        fill!(view(field, :, used), zero(eltype(field)))
    end
    return nothing
end