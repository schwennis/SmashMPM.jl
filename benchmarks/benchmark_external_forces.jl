# File: benchmarks/benchmark_external_forces.jl

"""
    add_external_force_benchmarks!(suite, backend; T=Float32, n_side=64, n_particles_sparse=25_000)

Register the external force benchmarks in `suite["external_forces"]`.
"""
function add_external_force_benchmarks!(
    suite::BenchmarkGroup,
    backend::Backend;
    T::Type{<:AbstractFloat}=Float32,
    n_side::Int=64,
    n_particles_sparse::Int=25_000,
)
    grp = suite["external_forces"] = BenchmarkGroup()
    dt  = T(1e-4)

    no_force = NoExternalForce()
    gravity  = ConstantGravity(SVector{3, T}(0.0, -9.81, 0.0))
    f_const  = SVector{3, T}(0.1, -9.81, 0.0)       # value of the constant VectorFieldForce

    # ------------------------------------------------------------------------ #
    #                                Dense Grid                                #
    # ------------------------------------------------------------------------ #
    N          = SVector{3, Int}(n_side, n_side, n_side)
    dense_grid = DenseGrid(T(1.0), N, zero(SVector{3, T}), 2, backend)
    populate_dense_grid!(dense_grid, backend)

    radial_dense    = RadialInvSquareForceField(T(-10.0), SVector{3, T}(n_side / 2, n_side / 2, n_side / 2))
    vec_force_dense = VectorFieldForce(SmashMPM._to_backend(backend, fill(f_const, Tuple(N))))

    dense = grp["DenseGrid_$(n_side)³"] = BenchmarkGroup()
    dense["NoExternalForce"] = @benchmarkable(
        sync_run!(apply_external_forces!, $backend, $no_force, $dense_grid, $dt)
    )
    dense["ConstantGravity"] = @benchmarkable(
        sync_run!(apply_external_forces!, $backend, $gravity, $dense_grid, $dt)
    )
    dense["RadialInvSquareForceField"] = @benchmarkable(
        sync_run!(apply_external_forces!, $backend, $radial_dense, $dense_grid, $dt)
    )
    dense["VectorFieldForce"] = @benchmarkable(
        sync_run!(apply_external_forces!, $backend, $vec_force_dense, $dense_grid, $dt)
    )
    # ConstantGravity is a fused broadcast, the field forces launch kernels
    is_cpu(backend) && expect_no_allocs!(dense, "NoExternalForce", "ConstantGravity")

    # ------------------------------------------------------------------------ #
    #                               Sparse Grid                                #
    # ------------------------------------------------------------------------ #
    # BoxDomain, so that the VectorFieldForce can be benchmarked as well
    sparse_grid, _, _, N_sparse = make_populated_sparse_grid(
        n_particles_sparse, backend; T=T, domain_type=:box, padding=2
    )
    radial_sparse    = RadialInvSquareForceField(T(-10.0), zero(SVector{3, T}))
    vec_force_sparse = VectorFieldForce(SmashMPM._to_backend(backend, fill(f_const, Tuple(N_sparse))))

    sparse = grp["SparseGrid_$(n_particles_sparse)p"] = BenchmarkGroup()
    sparse["NoExternalForce"] = @benchmarkable(
        sync_run!(apply_external_forces!, $backend, $no_force, $sparse_grid, $dt)
    )
    sparse["ConstantGravity"] = @benchmarkable(
        sync_run!(apply_external_forces!, $backend, $gravity, $sparse_grid, $dt)
    )
    sparse["RadialInvSquareForceField"] = @benchmarkable(
        sync_run!(apply_external_forces!, $backend, $radial_sparse, $sparse_grid, $dt)
    )
    sparse["VectorFieldForce"] = @benchmarkable(
        sync_run!(apply_external_forces!, $backend, $vec_force_sparse, $sparse_grid, $dt)
    )
    # Broadcast over views of the active tiles, the views must not escape
    is_cpu(backend) && expect_no_allocs!(sparse, "NoExternalForce", "ConstantGravity")

    return grp
end
