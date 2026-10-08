using JLD2


# ---------------------------------------------------------------------------- #
#                                  Parameters                                  #
# ---------------------------------------------------------------------------- #
_cons_params() = (
    T         = Float64,   # Float64 ist nötig, sonst wird die Rundung zum Problem
    dx        = 0.1,
    padding   = 6,
    ppc_1d    = 2,
    cfl       = 0.4,
    dt_max    = 1e-3,
    t_max     = 1.0,
    cube_size = 1.0,
    rho       = 1000.0,
    E         = 1e6,
    nu        = 0.3,
    omega     = 2.0,
    n_steps   = 200,
)

_cons_snapshot_dir()  = @__DIR__
_cons_snapshot_path() = joinpath(_cons_snapshot_dir(), "snapshot_000000.jld2")

# ---------------------------------------------------------------------------- #
#                        Create Snapshot if none exists                        #
# ---------------------------------------------------------------------------- #
function _cons_create_snapshot()
    p = _cons_params()
    T = p.T

    shape = RectangularPrism{T}(width = T(p.cube_size),
                                height = T(p.cube_size),
                                depth = T(p.cube_size))
    mat   = NeoHookean(ρ = T(p.rho), E = T(p.E), ν = T(p.nu))
    body  = Body(shape, zero(SVector{3, T}), SVector{3, T}(0, 0, p.omega), mat)

    setup = SimulationSetup(
        dx      = T(p.dx),
        t_max   = T(p.t_max),
        padding = p.padding,
        ppc_1d  = p.ppc_1d,
        cfl_number = T(p.cfl),
        dt_max  = T(p.dt_max),
        backend = CPU(),
    )

    model = build_mpm_model((body,), setup)

    mkpath(_cons_snapshot_dir())
    exporter = SmashMPM.JLD2Exporter(output_dir = _cons_snapshot_dir(),
                                     filename_prefix = "snapshot_")
    write_output(exporter, model, 0)

    isfile(_cons_snapshot_path()) || error("Failed to create snapshot for conservation tests.")
    return nothing
end

# ---------------------------------------------------------------------------- #
#                                 Measurements                                 #
# ---------------------------------------------------------------------------- #
function _cons_grid_totals(model)
    cpu = SmashMPM.model_to_cpu(model)
    st  = cpu.grid.state_read
    dx  = 1 / cpu.grid.inv_dx
    o   = cpu.grid.origin

    M = 0.0
    P = zero(SVector{3, Float64})
    L = zero(SVector{3, Float64})
    for I in CartesianIndices(st.mass)
        m = Float64(st.mass[I])
        m == 0.0 && continue
        p = SVector{3, Float64}(st.momentum.x[I], st.momentum.y[I], st.momentum.z[I])
        x = SVector{3, Float64}(o[1] + dx * (I[1] - 1),
                                o[2] + dx * (I[2] - 1),
                                o[3] + dx * (I[3] - 1))
        M += m
        P += p
        L += cross(x, p)
    end
    return (M = M, P = P, L = L)
end

# Total mass and momentum from the particles (v_p interpolated from the grid)
function _cons_particle_totals(model)
    cpu = SmashMPM.model_to_cpu(model)
    M = 0.0
    P = zero(SVector{3, Float64})
    for ps in cpu.particle_sets
        v, _ = SmashMPM.extract_velocities(cpu.grid, ps, cpu.shapefunction)
        m = ps.particles.mass
        for i in eachindex(v)
            M += m[i]
            P += m[i] * SVector{3, Float64}(v[i])
        end
    end
    return (M = M, P = P)
end

# ---------------------------------------------------------------------------- #
#                                     Test                                     #
# ---------------------------------------------------------------------------- #
@testset "Rotating Cube Conservation Test" begin
    p = _cons_params()

    isfile(_cons_snapshot_path()) || _cons_create_snapshot()
    @test isfile(_cons_snapshot_path())

    model = JLD2.load(_cons_snapshot_path(), "model")

    a     = p.cube_size
    ω     = p.omega
    tot0  = _cons_grid_totals(model)
    M0    = tot0.M
    P0    = tot0.P
    L0    = tot0.L

    v_ref = ω * a / 2
    P_ref = M0 * v_ref
    Lz_analytic = M0 * a^2 * ω / 6

    @testset "Initial State" begin
        @test M0 ≈ p.rho * a^3 rtol = 1e-2

        @test norm(P0) < 1e-8 * P_ref   # momentum should be zero except for rounding errors
        
        cpu0 = SmashMPM.model_to_cpu(model)
        ps   = cpu0.particle_sets[1].particles
        I_zz = sum(ps.mass[i] * (ps.pos.x[i]^2 + ps.pos.y[i]^2) for i in eachindex(ps.mass))
        dx   = 1 / cpu0.grid.inv_dx
        Lz_expected = ω * I_zz + M0 * ω * dx^2 / 2

        @test L0[3] ≈ Lz_expected rtol = 1e-8   # Particle + grid spin
        @test L0[3] ≈ Lz_analytic rtol = 5e-2   # Comparison with analytic value, but not exact due to discretization
        @test abs(L0[1]) < 1e-8 * abs(L0[3])
        @test abs(L0[2]) < 1e-8 * abs(L0[3])
    end

    @testset "Time Steps" begin
        # Calcilate derivation over time steps
        global max_dP    = 0.0   # |P(t) - P(0)|
        global max_dL    = 0.0   # |L(t) - L(0)| / |L(0)|
        global max_dM    = 0.0   # |M(t) - M(0)| / M(0)
        for _ in 1:p.n_steps
            step!(model)
            tot = _cons_grid_totals(model)
            max_dP = max(max_dP, norm(tot.P - P0))
            max_dL = max(max_dL, norm(tot.L - L0) / norm(L0))
            max_dM = max(max_dM, abs(tot.M - M0) / M0)
        end
    end

    @testset "Simulation really ran" begin
        @test model.t > 0
        particles = SmashMPM.model_to_cpu(model).particle_sets[1].particles
        @test any(F -> F != one(F), particles.F)   # Check if F changed (Which it should because of rotation)
    end

    @testset "Conservation of Mass" begin
        global max_dM
        @test max_dM < 1e-12
    end

    @testset "Conservation of Momentum" begin
        # Momentum should be conserved due to partition of unity
        global max_dP
        @test max_dP < 1e-8 * P_ref

        # test particles and grid
        part = _cons_particle_totals(model)
        grid = _cons_grid_totals(model)
        @test part.M ≈ grid.M rtol = 1e-12
        @test norm(part.P - grid.P) < 1e-8 * P_ref
    end

    @testset "Conservation of Angular Momentum" begin
        # APIC should conserve angular momentum exactly
        global max_dL
        @test max_dL < 1e-13

        # rotation axis should remain parallel to z-axis
        tot = _cons_grid_totals(model)
        @test abs(tot.L[1]) < 1e-6 * abs(tot.L[3])
        @test abs(tot.L[2]) < 1e-6 * abs(tot.L[3])
    end
end