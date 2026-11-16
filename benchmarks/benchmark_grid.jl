# File: benchmarks/benchmark_grid.jl

"""
    add_grid_benchmarks!(suite, backend; T=Float32, n_side=64, n_particles_sparse=25_000)

Register the grid benchmarks (reset, tile activation) in `suite["grid"]`.
The CFL reduction `max_wavespeed` works on the particles, see benchmark_solver.jl.
"""
function add_grid_benchmarks!(
    suite::BenchmarkGroup,
    backend::Backend;
    T::Type{<:AbstractFloat}=Float32,
    n_side::Int=64,
    n_particles_sparse::Int=25_000,
)
    grp = suite["grid"] = BenchmarkGroup()

    # ------------------------------------------------------------------------ #
    #                                Dense Grid                                #
    # ------------------------------------------------------------------------ #
    N          = SVector{3, Int}(n_side, n_side, n_side)
    dense_grid = DenseGrid(T(1.0), N, zero(SVector{3, T}), 2, backend)
    populate_dense_grid!(dense_grid, backend)

    dense = grp["DenseGrid_$(n_side)³"] = BenchmarkGroup()
    dense["grid_reset!"] = @benchmarkable(
        sync_run!(grid_reset!, $backend, $dense_grid)
    )
    # fill! on plain arrays, no kernel launch on the CPU
    is_cpu(backend) && expect_no_allocs!(dense, "grid_reset!")

    # ------------------------------------------------------------------------ #
    #                               Sparse Grid                                #
    # ------------------------------------------------------------------------ #
    sparse_grid, p_sets, spline, _ = make_populated_sparse_grid(
        n_particles_sparse, backend; T=T, domain_type=:open
    )

    sparse = grp["SparseGrid_$(n_particles_sparse)p"] = BenchmarkGroup()
    # activate_tiles! needs an empty write map before every evaluation (evals = 1)
    sparse["activate_tiles!"] = @benchmarkable(
        sync_run!(activate_tiles!, $backend, $sparse_grid, $p_sets, $spline),
        setup = (reset_tile_activation!($sparse_grid, $backend))
    )
    sparse["grid_reset_and_activate!"] = @benchmarkable(
        sync_run!(grid_reset!, $backend, $sparse_grid, $p_sets, $spline)
    )
    # Tile activation launches a kernel and reads the tile counter back to the host,
    # so no benchmark of this group is allocation-free

    return grp
end


"""
    add_grid_access_benchmarks!(suite, T=Float32; n_side=64, n_particles_sparse=25_000)

Register the grid access benchmarks in `suite["grid_access_T"]` (CPU, scalar group):
one particle's worth of grid lookups as done by the G2P (stencil cache, 27
`node_slot` calls, mass reads), DenseGrid vs. SparseGrid. Shows the cost of the
tile lookup and checks that the access layer (src/grids/grid_access.jl) is
allocation-free. Nanosecond scale, so it lives in the tuned scalar group.
"""
function add_grid_access_benchmarks!(
    suite::BenchmarkGroup,
    ::Type{T}=Float32;
    n_side::Int=64,
    n_particles_sparse::Int=25_000,
) where {T}
    grp     = suite["grid_access_$(T)"] = BenchmarkGroup()
    backend = CPU()
    spline  = QuadraticSpline()

    # DenseGrid: particle in the middle of the box
    N          = SVector{3, Int}(n_side, n_side, n_side)
    dense_grid = populate_dense_grid!(DenseGrid(T(1.0), N, zero(SVector{3, T}), 2, backend), backend)
    dense_acc  = SmashMPM.read_access(dense_grid)
    dense_pos  = SVector{3, T}(n_side / 2 + 0.3, n_side / 2 - 0.2, n_side / 2 + 0.1)

    # SparseGrid: at a particle position, so all stencil tiles are active
    sparse_grid, p_sets, _, _ = make_populated_sparse_grid(n_particles_sparse, backend; T=T, domain_type=:open)
    sparse_acc = SmashMPM.read_access(sparse_grid)
    sparse_pos = SmashMPM.grid_position(p_sets[1].particles.pos[1], sparse_grid.inv_dx, sparse_grid.origin)
    sparse_base = SmashMPM.get_support_base(spline, sparse_pos)

    grp["DenseGrid_stencil_lookup"]  = @benchmarkable stencil_lookup($dense_acc,  $spline, $dense_pos)
    grp["SparseGrid_stencil_lookup"] = @benchmarkable stencil_lookup($sparse_acc, $spline, $sparse_pos)
    grp["SparseGrid_stencil_cache"]  = @benchmarkable SmashMPM.stencil_cache($sparse_acc, $spline, $sparse_base)
    expect_no_allocs!(grp)

    return grp
end