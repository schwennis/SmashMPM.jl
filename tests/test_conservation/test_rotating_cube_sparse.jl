# ---------------------------------------------------------------------------- #
#              Rotating cube conservation test on the SparseGrid               #
# ---------------------------------------------------------------------------- #
# Same setup and tolerances as test_rotating_cube.jl (DenseGrid). The totals are
# summed over the nodes of the ACTIVE tiles of the read state. In addition, the
# run is compared step by step with a DenseGrid run and checked for lost writes.
# The model is built directly (no JLD2 snapshot needed).

# ---------------------------------------------------------------------------- #
#                                  Parameters                                  #
# ---------------------------------------------------------------------------- #
_scons_params() = (
    T         = Float64,   # Float64 is needed, otherwise rounding becomes a problem
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

function _scons_build_model(p; grid_type, domain=AutoDomain())
    T = p.T
    shape = RectangularPrism{T}(width  = T(p.cube_size),
                                height = T(p.cube_size),
                                depth  = T(p.cube_size))
    mat   = NeoHookean(ρ = T(p.rho), E = T(p.E), ν = T(p.nu))
    body  = Body(shape, zero(SVector{3, T}), SVector{3, T}(0, 0, p.omega), mat)

    setup = SimulationSetup(
        dx         = T(p.dx),
        t_max      = T(p.t_max),
        padding    = p.padding,
        ppc_1d     = p.ppc_1d,
        cfl_number = T(p.cfl),
        dt_max     = T(p.dt_max),
        grid_type  = grid_type,
        domain     = domain,
        backend    = CPU(),
    )
    return build_mpm_model((body,), setup)
end

# ---------------------------------------------------------------------------- #
#                                 Measurements                                 #
# ---------------------------------------------------------------------------- #
# Mass, momentum and angular momentum (about the world origin) of the read state
function _scons_accumulate(M, P, L, m, p, x)
    return M + m, P + p, L + cross(x, p)
end

function _scons_grid_totals(grid::SparseGrid)
    st        = grid.state_read
    tile_keys = grid.map_read.tile_keys
    M = 0.0
    P = zero(SVector{3, Float64})
    L = zero(SVector{3, Float64})
    for t in 1:grid.n_active_read, l in 1:SmashMPM.TILE_VOLUME
        m = Float64(st.mass[l, t])
        m == 0.0 && continue
        p = SVector{3, Float64}(st.momentum.x[l, t], st.momentum.y[l, t], st.momentum.z[l, t])
        i, j, k = SmashMPM.node_index(tile_keys[t], l)
        x = SVector{3, Float64}(SmashMPM.node_position(grid.origin, grid.inv_dx, i, j, k))
        M, P, L = _scons_accumulate(M, P, L, m, p, x)
    end
    return (M = M, P = P, L = L)
end

function _scons_grid_totals(grid::DenseGrid)
    st = grid.state_read
    dx = 1 / grid.inv_dx
    o  = grid.origin
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
        M, P, L = _scons_accumulate(M, P, L, m, p, x)
    end
    return (M = M, P = P, L = L)
end

# Total mass and momentum from the particles (v_p interpolated from the grid)
function _scons_particle_totals(model)
    M = 0.0
    P = zero(SVector{3, Float64})
    for ps in model.particle_sets
        v, _ = SmashMPM.extract_velocities(model.grid, ps, model.shapefunction)
        m = ps.particles.mass
        for i in eachindex(v)
            M += m[i]
            P += m[i] * SVector{3, Float64}(v[i])
        end
    end
    return (M = M, P = P)
end

# ---------------------------------------------------------------------------- #
#                                   Time loop                                  #
# ---------------------------------------------------------------------------- #
"""
Advance the sparse and the dense model in lockstep and collect the maximal
deviations: from the initial values (conservation) and between the two grids.
"""
function _scons_run!(sparse, dense, p, tot0, P_ref)
    max_dP = 0.0     # |P(t) - P(0)|
    max_dL = 0.0     # |L(t) - L(0)| / |L(0)|
    max_dM = 0.0     # |M(t) - M(0)| / M(0)
    diff_M = 0.0     # sparse vs dense
    diff_P = 0.0
    diff_L = 0.0
    max_missed = 0
    min_tiles  = typemax(Int)

    g = sparse.grid
    for _ in 1:p.n_steps
        dt_s = step!(sparse)
        dt_d = step!(dense)
        @assert dt_s ≈ dt_d "sparse and dense runs use different time steps"

        ts = _scons_grid_totals(sparse.grid)
        td = _scons_grid_totals(dense.grid)

        max_dP = max(max_dP, norm(ts.P - tot0.P))
        max_dL = max(max_dL, norm(ts.L - tot0.L) / norm(tot0.L))
        max_dM = max(max_dM, abs(ts.M - tot0.M) / tot0.M)

        diff_M = max(diff_M, abs(ts.M - td.M) / td.M)
        diff_P = max(diff_P, norm(ts.P - td.P) / P_ref)
        diff_L = max(diff_L, norm(ts.L - td.L) / norm(td.L))

        # lost P2G writes show up in the map that was the write map during g2p2g
        max_missed = max(max_missed, SmashMPM.missed_count(g.map_read),
                                     SmashMPM.missed_count(g.map_write))
        min_tiles  = min(min_tiles, g.n_active_read)
    end
    return (; max_dP, max_dL, max_dM, diff_M, diff_P, diff_L, max_missed, min_tiles)
end

# ---------------------------------------------------------------------------- #
#                                     Test                                     #
# ---------------------------------------------------------------------------- #
@testset "Rotating Cube Conservation Test (SparseGrid)" begin
    p = _scons_params()
    a = p.cube_size
    ω = p.omega

    for (label, domain) in (("OpenDomain", OpenDomain()), ("FittedBox", FittedBox()))
        @testset "$label" begin
            sparse = _scons_build_model(p; grid_type = SparseGrid, domain = domain)
            dense  = _scons_build_model(p; grid_type = DenseGrid)

            @test sparse.grid isa SparseGrid
            @test sparse.grid.domain isa (domain isa OpenDomain ? OpenDomain : BoxDomain)

            tot0 = _scons_grid_totals(sparse.grid)
            M0, P0, L0 = tot0.M, tot0.P, tot0.L

            v_ref = ω * a / 2
            P_ref = M0 * v_ref
            Lz_analytic = M0 * a^2 * ω / 6

            @testset "Initial State" begin
                @test M0 ≈ p.rho * a^3 rtol = 1e-2

                @test norm(P0) < 1e-8 * P_ref   # momentum should be zero except for rounding errors

                ps   = sparse.particle_sets[1].particles
                I_zz = sum(ps.mass[i] * (ps.pos.x[i]^2 + ps.pos.y[i]^2) for i in eachindex(ps.mass))
                dx   = 1 / sparse.grid.inv_dx
                Lz_expected = ω * I_zz + M0 * ω * dx^2 / 2

                @test L0[3] ≈ Lz_expected rtol = 1e-8   # Particle + grid spin
                @test L0[3] ≈ Lz_analytic rtol = 5e-2   # analytic value, not exact due to discretization
                @test abs(L0[1]) < 1e-8 * abs(L0[3])
                @test abs(L0[2]) < 1e-8 * abs(L0[3])

                # identical to the DenseGrid initial state
                td0 = _scons_grid_totals(dense.grid)
                @test M0 ≈ td0.M rtol = 1e-12
                @test norm(L0 - td0.L) < 1e-10 * norm(td0.L)

                # SparseGrid specific: tiles are active and nothing was lost
                @test sparse.grid.n_active_read > 0
                @test sparse.grid.n_active_write > 0
                @test SmashMPM.missed_count(sparse.grid.map_read) == 0
            end

            stats = _scons_run!(sparse, dense, p, tot0, P_ref)

            @testset "Simulation really ran" begin
                @test sparse.t > 0
                particles = sparse.particle_sets[1].particles
                @test any(F -> F != one(F), particles.F)   # F changed because of the rotation
            end

            @testset "Conservation of Mass" begin
                @test stats.max_dM < 1e-12
            end

            @testset "Conservation of Momentum" begin
                # Momentum is conserved due to the partition of unity
                @test stats.max_dP < 1e-8 * P_ref

                # particles and grid
                part = _scons_particle_totals(sparse)
                grid = _scons_grid_totals(sparse.grid)
                @test part.M ≈ grid.M rtol = 1e-12
                @test norm(part.P - grid.P) < 1e-8 * P_ref
            end

            @testset "Conservation of Angular Momentum" begin
                # APIC conserves angular momentum exactly
                @test stats.max_dL < 1e-13

                # rotation axis stays parallel to the z-axis
                tot = _scons_grid_totals(sparse.grid)
                @test abs(tot.L[1]) < 1e-6 * abs(tot.L[3])
                @test abs(tot.L[2]) < 1e-6 * abs(tot.L[3])
            end

            @testset "SparseGrid matches DenseGrid" begin
                @test stats.diff_M < 1e-12
                @test stats.diff_P < 1e-10
                @test stats.diff_L < 1e-11

                part_s = _scons_particle_totals(sparse)
                part_d = _scons_particle_totals(dense)
                @test part_s.M ≈ part_d.M rtol = 1e-12
                @test norm(part_s.P - part_d.P) < 1e-10 * P_ref
            end

            @testset "No lost writes, tiles stay active" begin
                @test stats.max_missed == 0
                @test stats.min_tiles > 0
            end
        end
    end
end