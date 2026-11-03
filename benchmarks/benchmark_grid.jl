# File: benchmarks/benchmark_grid.jl

"""
    add_grid_benchmarks!(suite, backend; T=Float32, n_side=64, n_particles_sparse=25_000)

Register the grid benchmarks (reductions, reset, tile activation) in `suite["grid"]`.
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
    dense["max_wavespeed"] = @benchmarkable(
        sync_run!(SmashMPM.max_wavespeed, $backend, $dense_grid)
    )
    dense["grid_reset!"] = @benchmarkable(
        sync_run!(grid_reset!, $backend, $dense_grid)
    )
    # maximum and fill! on plain arrays, no kernel launch on the CPU
    is_cpu(backend) && expect_no_allocs!(dense, "max_wavespeed", "grid_reset!")

    # ------------------------------------------------------------------------ #
    #                               Sparse Grid                                #
    # ------------------------------------------------------------------------ #
    sparse_grid, p_sets, spline, _ = make_populated_sparse_grid(
        n_particles_sparse, backend; T=T, domain_type=:open
    )

    sparse = grp["SparseGrid_$(n_particles_sparse)p"] = BenchmarkGroup()
    sparse["max_wavespeed"] = @benchmarkable(
        sync_run!(SmashMPM.max_wavespeed, $backend, $sparse_grid)
    )
    # activate_tiles! needs an empty write map before every evaluation (evals = 1)
    sparse["activate_tiles!"] = @benchmarkable(
        sync_run!(activate_tiles!, $backend, $sparse_grid, $p_sets, $spline),
        setup = (reset_tile_activation!($sparse_grid, $backend))
    )
    sparse["grid_reset_and_activate!"] = @benchmarkable(
        sync_run!(grid_reset!, $backend, $sparse_grid, $p_sets, $spline)
    )
    # Tile activation launches a kernel and reads the tile counter back to the host
    is_cpu(backend) && expect_no_allocs!(sparse, "max_wavespeed")

    return grp
end
