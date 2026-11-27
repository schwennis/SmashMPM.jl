# ---------------------------------------------------------------------------- #
#                      Transfer schemes: APIC / PolyPIC / heating              #
# ---------------------------------------------------------------------------- #
using Random

# Pressureless dust with an internal energy: σ = 0, so the grid sees no forces and
# the change of the kinetic energy is only due to the transfer. Ideal to check
# the heat bookkeeping (colliding dust blocks stick together, inelastic).
struct HeatDust{T} <: SmashMPM.AbstractMaterial
    ρ::T
end
struct HeatDustState{T} <: SmashMPM.AbstractMaterialState
    e::T
end
SmashMPM.initial_material_state(::HeatDust{T}) where {T} = HeatDustState(zero(T))
SmashMPM.material_model(::HeatDust, st::HeatDustState, F, C::SMatrix{3, 3, T, 9}, V0, m, dt, dx) where {T} =
    (zero(SMatrix{3, 3, T, 9}), st)
SmashMPM.soundspeed(::HeatDust, st::HeatDustState) = zero(st.e)
SmashMPM.has_internal_energy(::HeatDust) = true
SmashMPM.internal_energy(::HeatDust, st::HeatDustState) = st.e
SmashMPM.add_heat(::HeatDust, st::HeatDustState, Δe) = HeatDustState(st.e + Δe)


# Weights / bases of all 27 stencil nodes of a particle at grid position g
function _tr_stencil(g::SVector{3, Float64})
    spline = QuadraticSpline()
    hs     = SmashMPM.higher_stencil(PolyPIC(), spline, g)
    base   = SmashMPM.get_support_base(spline, g)
    nodes  = NamedTuple[]
    for di in 0:2, dj in 0:2, dk in 0:2
        nat = g - SVector(base[1] + di, base[2] + dj, base[3] + dk)
        ρ   = -nat
        push!(nodes, (N = shapefunction(spline, nat), ρ = ρ, φ = SmashMPM.higher_basis(hs, ρ, (di, dj, dk))))
    end
    return hs, nodes
end

@testset "PolyPIC basis" begin
    rng = Xoshiro(1)
    for _ in 1:20
        g = SVector{3}(10 .+ 3 .* rand(rng, 3))
        hs, nodes = _tr_stencil(g)
        t = SmashMPM.node_offset(g)
        @test all(-0.5 .<= t .< 0.5)
        @test hs.ν2 ≈ (1 .- 4 .* t .^ 2) .^ 2 .* (3 .- 4 .* t .^ 2) ./ 16
        # factored G = ρ² - aρ - ¼ with a = t(1 - 4t²), nodes ρ = -1-t, -t, 1-t
        for k in 1:3, d in 1:3
            ρ = d - 2 - t[k]
            @test hs.G[k, d] ≈ ρ^2 - SmashMPM.poly_a(t[k]) * ρ - 0.25 atol = 1e-14
        end

        # Gram matrix of all 10 modes: diagonal, norms ν₀ = 1, ν₁ = ¼, ν(xy) = 1/16, ν₂(t_k)
        Φ = [vcat(1.0, n.ρ, n.φ) for n in nodes]
        Gram = sum(n.N * φ * φ' for (n, φ) in zip(nodes, Φ))
        expected = vcat(1.0, fill(0.25, 3), fill(1 / 16, 3), hs.ν2)
        @test Gram ≈ Diagonal(expected) atol = 1e-14
    end
end

@testset "G modes near a cell face (Float32)" begin
    # Particle at t = ½ - s: the G modes are O(s²) at the heavy nodes. With the
    # factored formulas the projection stays accurate to a few eps of Float32;
    # ρ² - aρ - ¼ directly had an error ~ eps/s² (≈ 1 at s = 1e-4).
    spline = QuadraticSpline()
    rng    = Xoshiro(3)
    for s in (1f-2, 1f-3, 1f-4), axis in 1:3
        g  = SVector{3, Float32}(7.3f0, 4.81f0, 5.07f0)
        g  = setindex(g, 10.5f0 - s, axis)
        hs = SmashMPM.higher_stencil(PolyPIC(), spline, g)
        base = SmashMPM.get_support_base(spline, g)
        c0 = SVector{3, Float32}(randn(rng, 3)); Cl = SMatrix{3, 3, Float32}(randn(rng, 3, 3))
        ch = SMatrix{3, 6, Float32}(randn(rng, 3, 6))
        b  = SmashMPM.zero_higher(PolyPIC(), Float32)
        for di in 0:2, dj in 0:2, dk in 0:2
            ρ = -(g - SVector(base[1] + di, base[2] + dj, base[3] + dk))
            φ = SmashMPM.higher_basis(hs, ρ, (di, dj, dk))
            N = shapefunction(spline, -ρ)
            b = SmashMPM.accumulate_higher(b, N, c0 + Cl * ρ + ch * φ, φ)
        end
        c, _, _ = SmashMPM.finalize_higher(PolyPIC(), b, hs, hs)
        @test hs.ν2[axis] > 0
        @test maximum(abs, c[:, 3 + axis] - ch[:, 3 + axis]) < 1f-3 * maximum(abs, ch)
    end
end

@testset "PolyPIC projection and minimal damping" begin
    rng    = Xoshiro(2)
    spline = QuadraticSpline()
    dx     = 0.1
    g      = SVector(7.3, 4.81, 5.07)
    hs, nodes = _tr_stencil(g)

    # Grid velocity field inside the span of the 10 modes
    c0 = SVector{3}(randn(rng, 3))
    Cl = SMatrix{3, 3}(randn(rng, 3, 3))       # coefficients of ρ (units of Δx)
    ch = SMatrix{3, 6}(randn(rng, 3, 6))       # xy, xz, yz, Gx, Gy, Gz
    v_I = [c0 + Cl * n.ρ + ch * n.φ for n in nodes]

    # G2P exactly as in the kernel
    vel = zero(SVector{3, Float64}); B = zero(SMatrix{3, 3, Float64, 9})
    b   = SmashMPM.zero_higher(PolyPIC(), Float64)
    for (n, v) in zip(nodes, v_I)
        vel += n.N * v
        B   += SmashMPM.B_update(spline, n.N, n.ρ * dx, v)
        b    = SmashMPM.accumulate_higher(b, n.N, v, n.φ)
    end
    C = B * SmashMPM.M_inv(spline, 1 / dx)

    # Particle at rest: no damping, the field is recovered exactly
    c, e_high, e_damp = SmashMPM.finalize_higher(PolyPIC(), b, hs, hs)
    @test vel ≈ c0
    @test C * dx ≈ Cl
    @test c ≈ ch
    @test e_damp == 0
    @test e_high ≈ 0.5 * (sum(abs2, ch[:, 1:3]) / 16 + sum(k -> hs.ν2[k] * sum(abs2, ch[:, 3 + k]), 1:3))

    # Kinetic energy of the field = energy of the modes (Parseval, the field is in the span)
    E_grid = 0.5 * sum(n.N * dot(v, v) for (n, v) in zip(nodes, v_I))
    @test E_grid ≈ 0.5 * (dot(c0, c0) + 0.25 * sum(abs2, Cl)) + e_high

    # Particle moves: the deposited G energy ν₂' |c'|² never exceeds ν₂ |c|²
    for g_new in (SVector(7.0, 4.5 + 1e-3, 5.2), SVector(7.45, 4.99, 5.07), g .+ 0.37)
        hs_new = SmashMPM.higher_stencil(PolyPIC(), spline, g_new)
        c2, e2, d2 = SmashMPM.finalize_higher(PolyPIC(), b, hs, hs_new)
        @test e2 ≈ e_high
        @test c2[:, 1:3] ≈ ch[:, 1:3]               # bilinear modes are never damped
        for k in 1:3
            ν, ν′ = hs.ν2[k], hs_new.ν2[k]
            e_in  = 0.5 * ν  * sum(abs2, ch[:, 3 + k])
            e_out = 0.5 * ν′ * sum(abs2, c2[:, 3 + k])
            @test e_out <= e_in * (1 + 1e-12)
            ν′ >= ν && @test e_out ≈ e_in            # energy neutral, minimal damping
            ν′ <  ν && @test c2[:, 3 + k] ≈ ch[:, 3 + k]   # no damping needed
        end
        # damping loss = extracted - deposited energy of the G modes
        @test d2 ≈ sum(k -> 0.5 * max(hs.ν2[k] - hs_new.ν2[k], 0) * sum(abs2, ch[:, 3 + k]), 1:3)
    end

    @test_throws ErrorException SmashMPM.check_transfer(PolyPIC(), nothing)
end


# ---------------------------------------------------------------------------- #
#                         Full simulations: helpers                            #
# ---------------------------------------------------------------------------- #
function _tr_grid_totals(model)
    cpu  = SmashMPM.model_to_cpu(model)
    grid = cpu.grid isa SparseGrid ? to_dense(cpu.grid) : cpu.grid
    st, dx, o = grid.state_read, 1 / grid.inv_dx, grid.origin
    M = 0.0; P = zero(SVector{3, Float64}); L = zero(SVector{3, Float64})
    for I in CartesianIndices(st.mass)
        m = st.mass[I]
        m == 0 && continue
        p = SVector(st.momentum.x[I], st.momentum.y[I], st.momentum.z[I])
        x = o + dx * SVector(I[1] - 1, I[2] - 1, I[3] - 1)
        M += m; P += p; L += cross(x, p)
    end
    return (M = M, P = P, L = L)
end

function _tr_rotating_cube(transfer, grid_type)
    T     = Float64
    shape = RectangularPrism{T}(width = 0.6, height = 0.5, depth = 0.4, center = SVector(0.03, -0.02, 0.01))
    mat   = NeoHookean(ρ = 1000.0, E = 1e6, ν = 0.3)
    body  = Body(shape, SVector(0.3, -0.1, 0.2), SVector(0.5, -1.0, 2.0), mat)
    setup = SimulationSetup(dx = 0.1, t_max = 10.0, padding = 4, buffer_width = SVector(4, 4, 4),
                            dt_max = 1e-3, grid_type = grid_type, transfer = transfer)
    return build_mpm_model((body,), setup)
end

function _tr_dust_collision(transfer, grid_type; T = Float64)
    mat = HeatDust(T(1000))
    a   = RectangularPrism{T}(width = T(0.5), height = T(0.5), depth = T(0.4), center = SVector{3, T}(-0.3, 0.05, 0))
    b   = RectangularPrism{T}(width = T(0.4), height = T(0.4), depth = T(0.4), center = SVector{3, T}(0.27, -0.05, 0.02))
    bodies = (Body(a, SVector{3, T}(1, 0.1, 0), SVector{3, T}(0, 0, 2), mat),
              Body(b, SVector{3, T}(-1.5, 0, 0.2), SVector{3, T}(1, 0, 0), mat))
    setup = SimulationSetup(dx = T(0.1), t_max = T(10), padding = 4, buffer_width = SVector(5, 5, 5),
                            dt_max = T(0.01), cfl_number = T(0.4), grid_type = grid_type, transfer = transfer)
    return build_mpm_model(bodies, setup)
end


@testset "PolyPIC conservation ($(nameof(G)))" for G in (DenseGrid, SparseGrid)
    for transfer in (PolyPIC(), PolyPIC(heating = true))
        model = _tr_rotating_cube(transfer, G)
        tot0  = _tr_grid_totals(model)
        max_dP = 0.0; max_dL = 0.0; max_dM = 0.0
        for _ in 1:100
            step!(model)
            tot = _tr_grid_totals(model)
            max_dM = max(max_dM, abs(tot.M - tot0.M) / tot0.M)
            max_dP = max(max_dP, norm(tot.P - tot0.P) / norm(tot0.P))
            max_dL = max(max_dL, norm(tot.L - tot0.L) / norm(tot0.L))
        end
        @test model.t > 0
        @test max_dM < 1e-12
        @test max_dP < 1e-12
        @test max_dL < 1e-12      # higher modes carry no angular momentum
    end

    # Elastic + kinetic energy of the rotating cube: PolyPIC loses less than APIC
    E_rel(transfer) = begin
        model = _tr_rotating_cube(transfer, G)
        E0 = energy_budget(model).total
        for _ in 1:100
            step!(model)
        end
        energy_budget(model).total / E0
    end
    e_apic, e_poly = E_rel(APIC()), E_rel(PolyPIC())
    @test e_apic < 1 + 1e-3 && e_poly < 1 + 1e-3
    @test 1 - e_poly < 1 - e_apic
end


@testset "Heat bookkeeping ($(nameof(G)))" for G in (DenseGrid, SparseGrid)
    for scheme in (APIC, PolyPIC)
        # Without heating: the collision dissipates kinetic energy
        model = _tr_dust_collision(scheme(), G)
        E0    = energy_budget(model)
        for _ in 1:120
            step!(model)
        end
        E_cold = energy_budget(model)
        @test E_cold.internal == 0
        @test E_cold.pending == 0
        loss = (E0.total - E_cold.total) / E0.total
        @test loss > 0.1

        # With heating: kinetic + internal + pending is conserved by the transfer
        model = _tr_dust_collision(scheme(heating = true), G)
        E0    = energy_budget(model)
        max_err = 0.0
        for _ in 1:120
            step!(model)
            max_err = max(max_err, abs(energy_budget(model).total - E0.total) / E0.total)
        end
        E_hot = energy_budget(model)
        @test max_err < 1e-12
        @test E_hot.internal > 0.1 * E0.total
        # same dynamics as without heating (dust: e does not feed back)
        @test E_hot.kinetic ≈ E_cold.kinetic rtol = 1e-10
        # all particles were heated, none cooled (up to rounding)
        for ps in SmashMPM.model_to_cpu(model).particle_sets
            @test minimum(st -> st.e, ps.particles.mat_state) > -1e-12 * E0.total
        end
    end
end


@testset "PolyPIC dissipates less than APIC" begin
    # Same collision, same steps: the 10-mode transfer keeps more kinetic energy
    lost(transfer) = begin
        model = _tr_dust_collision(transfer, DenseGrid)
        E0 = energy_budget(model).total
        for _ in 1:40
            step!(model)
        end
        E0 - energy_budget(model).total
    end
    @test lost(PolyPIC()) < lost(APIC())
end


@testset "Heating with SolidMaterial" begin
    T     = Float64
    mat   = Aluminum(T)
    a     = RectangularPrism{T}(width = 2e-3, height = 2e-3, depth = 2e-3, center = SVector(-1.2e-3, 0, 0))
    b     = RectangularPrism{T}(width = 2e-3, height = 2e-3, depth = 2e-3, center = SVector(1.2e-3, 0, 0))
    bodies = (Body(a, SVector{3, T}(300, 0, 0), zero(SVector{3, T}), mat),
              Body(b, SVector{3, T}(-300, 0, 0), zero(SVector{3, T}), mat))
    run(transfer) = begin
        setup = SimulationSetup(dx = T(2.5e-4), t_max = T(1), padding = 4, buffer_width = SVector(4, 4, 4),
                                dt_max = T(1e-6), grid_type = SparseGrid, transfer = transfer)
        model = build_mpm_model(bodies, setup)
        E0 = energy_budget(model)
        for _ in 1:60
            step!(model)
        end
        E = energy_budget(model)
        (E.total / E0.total, E.internal / E0.total, E.elastic / E0.total)
    end
    apic_cold, apic_hot, poly_cold, poly_hot = run.((APIC(), APIC(heating = true), PolyPIC(), PolyPIC(heating = true)))
    # The impact converts the kinetic energy into elastic and internal energy
    for r in (apic_cold, apic_hot, poly_cold, poly_hot)
        @test all(isfinite, r)
        @test r[2] > 0.05 && r[3] > 0.2
    end
    # APIC without heating loses energy; with heating the budget closes up to the
    # leapfrog fluctuation at the contact
    @test apic_cold[1] < 0.95
    @test abs(apic_hot[1] - 1) < 0.02
    @test abs(poly_hot[1] - 1) < 0.03
    @test apic_hot[2] > apic_cold[2]
    @test poly_hot[2] > poly_cold[2]
end

@testset "Heating is Galilean invariant (Float32)" begin
    # Rigidly translating block: physically no heat. The heat terms are evaluated
    # relative to reference velocities, so ½|v|² never has to cancel. Before, a
    # Float32 block at 1000 m/s got ±6 J/kg of noise in 100 steps (now ~1e-4).
    T   = Float32
    a   = RectangularPrism{T}(width = T(2e-3), height = T(2e-3), depth = T(2e-3), center = zero(SVector{3, T}))
    for G in (DenseGrid, SparseGrid), tr in (APIC(heating = true), PolyPIC(heating = true))
        body  = Body(a, SVector{3, T}(1000, 300, 0), zero(SVector{3, T}), Aluminum(T))
        setup = SimulationSetup(dx = T(2.5e-4), t_max = T(1), padding = 4, buffer_width = SVector(8, 8, 8),
                                dt_max = T(1e-6), grid_type = G, transfer = tr)
        model = build_mpm_model((body,), setup)
        @test model.dt_previous isa T
        for _ in 1:40
            step!(model)
        end
        e = [st.e for st in SmashMPM.model_to_cpu(model).particle_sets[1].particles.mat_state]
        @test maximum(abs, e) < T(1e-2)        # J/kg, ½v² = 5.4e5 J/kg
    end
end

@testset "Heating warning for materials without internal energy" begin
    body  = Body(RectangularPrism{Float64}(width = 0.3, height = 0.3, depth = 0.3),
                 zero(SVector{3, Float64}), zero(SVector{3, Float64}), NeoHookean(ρ = 1000.0, E = 1e6, ν = 0.3))
    setup = SimulationSetup(dx = 0.1, t_max = 1.0, transfer = APIC(heating = true))
    @test_logs (:warn, r"no internal energy") match_mode = :any build_mpm_model((body,), setup)
end

# ---------------------------------------------------------------------------- #
#                       Aluminium impact: time integration                     #
# ---------------------------------------------------------------------------- #
function _tr_al_impact(transfer; cfl = 0.4, G = SparseGrid)
    T = Float64
    a = RectangularPrism{T}(width = 2e-3, height = 2e-3, depth = 2e-3, center = SVector(-1.2e-3, 0, 0))
    b = RectangularPrism{T}(width = 2e-3, height = 2e-3, depth = 2e-3, center = SVector(1.2e-3, 0, 0))
    bodies = (Body(a, SVector{3, T}(300, 0, 0), zero(SVector{3, T}), Aluminum(T)),
              Body(b, SVector{3, T}(-300, 0, 0), zero(SVector{3, T}), Aluminum(T)))
    setup = SimulationSetup(dx = T(2.5e-4), t_max = T(1), padding = 4, buffer_width = SVector(4, 4, 4),
                            dt_max = T(1e-6), cfl_number = T(cfl), grid_type = G, transfer = transfer)
    return build_mpm_model(bodies, setup)
end

@testset "PolyPIC(heating) overshoot is the O(Δt) time integration error" begin
    # With heating the transfer conserves the energy exactly, the remaining change
    # of the total at the same physical time comes from the time integration and
    # halves with Δt. It oscillates in time (≈ -2 % at 0.43 µs for CFL 0.4).
    dE(cfl) = begin
        model = _tr_al_impact(PolyPIC(heating = true); cfl = cfl)
        E0    = energy_budget(model).total
        for _ in 1:round(Int, 31 * 0.4 / cfl)
            step!(model)
        end
        energy_budget(model).total / E0 - 1
    end
    d4, d2 = dE(0.4), dE(0.2)
    @test abs(d4) > 0.005
    @test 1.6 < d4 / d2 < 2.6
end

@testset "Allocations of g2p2g!" begin
    for tr in (APIC(), APIC(heating = true), PolyPIC(), PolyPIC(heating = true))
        model = _tr_rotating_cube(tr, SparseGrid)
        step!(model); step!(model)
        dt = model.dt_previous
        SmashMPM.g2p2g!(model, dt)
        @test (@allocated SmashMPM.g2p2g!(model, dt)) < 50_000      # task overhead only
    end
end

@testset "Multithreaded runs only differ by rounding" begin
    # With several threads the atomic adds come in a different order every run.
    # That must only cause rounding differences: a discontinuous branch in the
    # update (as the ∂p/∂e jump of Tillotson at e = 0 before) amplified them to
    # ~1e-8 Δx within a few steps at the contact. Now: ~1e-14 Δx after 240 steps.
    if Threads.nthreads() == 1
        @test true
    else
        for tr in (APIC(), PolyPIC())
            m1, m2 = _tr_al_impact(tr), _tr_al_impact(tr)
            d = 0.0
            for _ in 1:30
                step!(m1); step!(m2)
                d = max(d, maximum(s -> maximum(norm.(m1.particle_sets[s].particles.pos .-
                                                      m2.particle_sets[s].particles.pos)), 1:2))
            end
            @test d < 1e-10 * 2.5e-4
        end
    end
end
