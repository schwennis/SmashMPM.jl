# File: benchmarks/benchmark_solver.jl

"""
    add_solver_benchmarks!(suite, backend; T=Float32, particle_counts=[10_000, 50_000])

Register the solver benchmarks in `suite["solver"]`: the isolated G2P2G kernel and
a full time step for every combination of grid type and material.
"""
function add_solver_benchmarks!(
    suite::BenchmarkGroup,
    backend::Backend;
    T::Type{<:AbstractFloat}=Float32,
    particle_counts::Vector{Int}=[10_000, 50_000],
)
    grp = suite["solver"] = BenchmarkGroup()

    mat_simple = NeoHookean(ρ=T(1000.0), E=T(1e6), ν=T(0.3))
    mat_solid  = Basalt(T; eos=:tillotson, elasticity=:hypoelastic, viscosity=BulkViscosity{T}())
    gravity    = ConstantGravity(SVector{3, T}(0.0, -9.81, 0.0))
    noslip     = NoSlipBoundary()

    grid_configs = (
        # label              grid type   domain        boundary condition
        ("DenseGrid",        DenseGrid,  AutoDomain(), noslip),
        ("SparseGrid_Open",  SparseGrid, OpenDomain(), NullBoundaryCondition()),
        ("SparseGrid_Box",   SparseGrid, FittedBox(),  noslip),
    )
    materials = (("NeoHookean", mat_simple), ("Basalt_Tillotson", mat_solid))

    for n_target in particle_counts
        sub = grp["N_$(n_target)"] = BenchmarkGroup()

        for (grid_label, grid_type, domain, bc) in grid_configs, (mat_label, mat) in materials
            model, _ = build_benchmark_model(
                n_target, backend;
                T=T,
                grid_type=grid_type,
                domain=domain,
                material=mat,
                boundary_condition=bc,
                external_force=gravity,
            )
            backup  = capture_particle_backup(model)
            dt_tiny = T(1e-7)       # particles hardly move, so the active tiles stay valid between restores

            case = sub["$(grid_label)_$(mat_label)"] = BenchmarkGroup()

            # isolated G2P2G kernel
            case["g2p2g!"] = @benchmarkable(
                g2p2g!($model, $dt_tiny),
                setup = (restore_particles!($model, $backup))
            )

            # full time step: CFL + G2P2G + forces + BCs + grid reset / tile activation
            case["step!"] = @benchmarkable(
                step!($model),
                setup = (restore_particles!($model, $backup))
            )
            # Kernel launches allocate on every backend, so allocations are not checked here
        end
    end

    return grp
end
