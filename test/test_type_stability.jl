# ---------------------------------------------------------------------------- #
#                   Type stability (CPU and GPU precision)                     #
# ---------------------------------------------------------------------------- #
# Everything that runs inside a kernel must be fully inferable: on the GPU a
# runtime dispatch does not compile, on the CPU it costs an allocation per
# particle and step. Therefore JET is applied directly to the DEVICE functions
# (material_model, g2p2g_particle!, initial_p2g_particle!), in Float64 and Float32:
#   @test_opt   no runtime dispatch / failed optimization
#   @test_call  no possible MethodError etc., also in branches the test never runs
#
# Host functions that launch KernelAbstractions kernels: inside KA's CPU thread
# scheduler JET reports false positives (UndefVar in `sync_end`, inference
# recursion limits). For the grid operations JET is therefore restricted to
# SmashMPM's own code (target_modules); step! and build_mpm_model are checked
# with @inferred and allocation bounds.

@testset "material_model: $name ($T)" for T in (Float64, Float32), (name, mat_ref) in material_zoo(T)
    # body-level initialization: the material of the set may carry sampled data (flaws)
    mat, states = initial_material_states(mat_ref, Xoshiro(1), 100, T(1e-3))
    st   = states[1]
    F    = SMatrix{3, 3, T, 9}(1.01, 0.002, 0, 0.001, 0.99, 0, 0, 0, 1)
    C    = SMatrix{3, 3, T, 9}(-10, 1, 0, 2, 5, 0, 0, 0, -3)
    args = (mat, st, F, C, T(1e-9), T(2.7e-6), T(1e-7), T(1e-3))

    # The particle arrays store ONE state type (taken from the deterministic
    # initial state): neither the random initialization nor the update may change it.
    @test typeof(st) === typeof(initial_material_state(mat_ref))
    @test typeof(st) === typeof(initial_material_state(mat))
    σ, st_new = @inferred material_model(args...)
    @test typeof(st_new) === typeof(st)
    @test σ isa SMatrix{3, 3, T, 9}
    @test_opt material_model(args...)
    @test_call material_model(args...)
    @test allocs(material_model, args...) == 0
end


# Most complex material: Tillotson + hyperelasticity + J2 + Grady-Kipp + bulk viscosity
kernel_material(T) = Basalt(T; elasticity = :hyperelastic, viscosity = BulkViscosity{T}(),
                            plasticity = J2Plasticity(σ_y0 = T(1e8), H = T(1e9)))

function kernel_model(grid_type, transfer, T)
    body = Body(RectangularPrism{T}(width = T(0.4), height = T(0.3), depth = T(0.3)),
                SVector{3, T}(1, 0, 0), SVector{3, T}(0, 0, 2), kernel_material(T))
    return build_mpm_model((body,), SimulationSetup(dx = T(0.1), t_max = T(1), grid_type = grid_type,
                                                    transfer = transfer))
end

@testset "particle kernels: $(nameof(G)), $(nameof(typeof(tr)))(heating=$(S.transfer_heating(tr))), $T" for
        G in (DenseGrid, SparseGrid), tr in (APIC(), PolyPIC(heating = true)), T in (Float64, Float32)

    model = kernel_model(G, tr, T)
    ps, g = model.particle_sets[1], model.grid

    # G2P2G of one particle (exactly the code a GPU thread runs)
    args = (1, ps.particles, ps.material, S.read_access(g), S.write_access(g),
            g.origin, g.inv_dx, model.shapefunction, model.transfer, T(1e-7))
    @test_opt S.g2p2g_particle!(args...)
    @test_call S.g2p2g_particle!(args...)
    @test allocs(S.g2p2g_particle!, args...) == 0

    # initial P2G of one particle
    pos  = collect(ps.particles.pos)
    args = (1, S.write_access(g), pos, zero(pos), zero(SMatrix{3, 3, T, 9}), collect(ps.particles.mass),
            g.origin, g.inv_dx, model.shapefunction)
    @test_opt S.initial_p2g_particle!(args...)
    @test_call S.initial_p2g_particle!(args...)
end


# Grid operations of every step. Besides JET: their allocations must not grow with
# the grid (8x the nodes). KA spawns up to nthreads tasks per launch, hence the
# small allowance per thread.
@testset "grid operations: $(nameof(G))" for G in (DenseGrid, SparseGrid)
    field(n) = fill(V3(1, 2, 3), n, n, n)
    ops = ("grid_reset!"      => (g, n) -> (S.grid_reset!, (g,)),
           "grid_kick!"       => (g, n) -> (S.grid_kick!, (g, 0.1)),
           "ConstantGravity"  => (g, n) -> (apply_external_forces!, (ConstantGravity(V3(0, -9.81, 0)), g, 0.1)),
           "RadialInvSquare"  => (g, n) -> (apply_external_forces!, (RadialInvSquareForceField(-3.0, zero(V3)), g, 0.1)),
           "VectorFieldForce" => (g, n) -> (apply_external_forces!, (VectorFieldForce(field(n)), g, 0.1)),
           "NoSlipBoundary"   => (g, n) -> (apply_boundary_condition!, (NoSlipBoundary(), g)))

    @testset "$name" for (name, op) in ops
        f, args   = op(box_grid(G, 16), 16)
        _, args32 = op(box_grid(G, 32), 32)
        @test_opt target_modules = (SmashMPM,) f(args...)
        @test_call target_modules = (SmashMPM,) f(args...)
        grows = allocs(f, args32...) > allocs(f, args...) + 2048 * Threads.nthreads()
        @test allocs(f, args32...) <= allocs(f, args...) + 2048 * Threads.nthreads()
    end
end

@testset "CFL reduction over particle sets" begin
    sets = ((particles = (wave_speed = [1.0, 15.5, 2.0],),),
            (particles = (wave_speed = Float64[],),),               # empty set must not throw
            (particles = (wave_speed = [3.0],),))
    @test @inferred(S.max_wavespeed(sets)) == 15.5
    @test S.max_wavespeed(((particles = (wave_speed = Float64[],),),)) == 0
    @test_opt S.max_wavespeed(sets)
end


# Two bodies with different material types: the particle sets form a
# heterogeneous tuple, every `map` / `foreach` over it must be unrolled.
hetero_bodies(T) =
    (Body(Sphere(T(0.3), SVector{3, T}(-0.4, 0, 0)), SVector{3, T}(1, 0, 0), SVector{3, T}(0, 0, 2),
          NeoHookean(ρ = T(1000), E = T(1e6), ν = T(0.3))),
     Body(RectangularPrism{T}(width = T(0.4), height = T(0.3), depth = T(0.3), center = SVector{3, T}(0.4, 0, 0)),
          zero(SVector{3, T}), zero(SVector{3, T}), Aluminum(T)))
hetero_setup(G, T; ppc = 2) = SimulationSetup(dx = T(0.1), t_max = T(1), ppc_1d = ppc, grid_type = G, transfer = PolyPIC())

@testset "host side: $(nameof(G)), $T" for G in (DenseGrid, SparseGrid), T in (Float64, Float32)
    bodies = hetero_bodies(T)
    model  = @inferred build_mpm_model(bodies, hetero_setup(G, T))
    @test model.particle_sets[1].material isa NeoHookean{T}
    @test eltype(model.particle_sets[2].particles.mat_state) <: SolidMaterialState{T}

    # precision is never promoted: grid, particles and clock stay in T
    @test eltype(model.grid.state_read.mass) === T
    @test eltype(model.particle_sets[1].particles.F) === SMatrix{3, 3, T, 9}
    @test model.t isa T && model.dt_previous isa T

    @test @inferred(step!(model)) isa T
    @test @inferred(S.courant_timestep(model)) isa T
    # diagnostic only (not inferable over a heterogeneous tuple), but must stay in T
    @test all(x -> x isa T, energy_budget(model))

    # step! allocates per kernel launch, not per particle: 3.4x the particles on the
    # same grid allocate the same. (KA spawns up to nthreads tasks per launch, hence
    # the small allowance per thread; the kernels themselves are allocation free, see above.)
    a4 = build_mpm_model(bodies, hetero_setup(G, T; ppc = 4))
    a6 = build_mpm_model(bodies, hetero_setup(G, T; ppc = 6))
    @test allocs(step!, a6) <= allocs(step!, a4) + 8192 * Threads.nthreads()
end