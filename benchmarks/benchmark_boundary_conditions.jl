# File: benchmarks/benchmark_boundary_conditions.jl

"""
    add_boundary_condition_benchmarks!(suite, backend; T=Float32, n_side=64, n_particles_sparse=25_000)

Register the boundary condition benchmarks in `suite["boundary_conditions"]`.
"""
function add_boundary_condition_benchmarks!(
    suite::BenchmarkGroup,
    backend::Backend;
    T::Type{<:AbstractFloat}=Float32,
    n_side::Int=64,
    n_particles_sparse::Int=25_000,
)
    grp = suite["boundary_conditions"] = BenchmarkGroup()

    null_bc   = NullBoundaryCondition()
    noslip_bc = NoSlipBoundary()

    # ------------------------------------------------------------------------ #
    #                                Dense Grid                                #
    # ------------------------------------------------------------------------ #
    N          = SVector{3, Int}(n_side, n_side, n_side)
    dense_grid = DenseGrid(T(1.0), N, zero(SVector{3, T}), 4, backend)
    populate_dense_grid!(dense_grid, backend)

    dense = grp["DenseGrid_$(n_side)³"] = BenchmarkGroup()
    dense["NullBoundaryCondition"] = @benchmarkable(
        sync_run!(apply_boundary_condition!, $backend, $null_bc, $dense_grid)
    )
    dense["NoSlipBoundary"] = @benchmarkable(
        sync_run!(apply_boundary_condition!, $backend, $noslip_bc, $dense_grid)
    )
    # The null condition is a no-op, NoSlipBoundary launches a kernel
    is_cpu(backend) && expect_no_allocs!(dense, "NullBoundaryCondition")

    # ------------------------------------------------------------------------ #
    #                               Sparse Grid                                #
    # ------------------------------------------------------------------------ #
    # NoSlipBoundary needs walls, i.e. a BoxDomain
    sparse_grid, _, _, _ = make_populated_sparse_grid(
        n_particles_sparse, backend; T=T, domain_type=:box, padding=4
    )

    sparse = grp["SparseGrid_$(n_particles_sparse)p"] = BenchmarkGroup()
    sparse["NullBoundaryCondition"] = @benchmarkable(
        sync_run!(apply_boundary_condition!, $backend, $null_bc, $sparse_grid)
    )
    sparse["NoSlipBoundary"] = @benchmarkable(
        sync_run!(apply_boundary_condition!, $backend, $noslip_bc, $sparse_grid)
    )
    is_cpu(backend) && expect_no_allocs!(sparse, "NullBoundaryCondition")

    return grp
end
