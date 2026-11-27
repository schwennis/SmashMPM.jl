# ---------------------------------------------------------------------------- #
#                   Full simulations: physics of the whole solver               #
# ---------------------------------------------------------------------------- #
#   setup        particle generation, initial P2G incl. half kick (exact formulas)
#   time         kick-drift-kick is exact for constant acceleration, ends at t_max
#   conservation mass, momentum and angular momentum to rounding, for APIC and
#                PolyPIC, on DenseGrid / SparseGrid(open) / SparseGrid(box); the
#                three grids give the same trajectories
#   energy       elastic oscillation (stress works against the strain energy),
#                heat bookkeeping closes the energy budget, Galilean invariance,
#                remaining error is the O(Δt) time integration error

# ---------------------------------------------------------------------------- #
#                                    Setup                                     #
# ---------------------------------------------------------------------------- #
@testset "Particle generation" begin
    ρ, h  = 2000.0, 0.02
    v0, ω = V3(1, 0, 0.5), V3(0.3, -1, 2)
    shapes = (Sphere(0.3, V3(0.1, -0.2, 0.3))                                         => 4 / 3 * π * 0.3^3,
              RectangularPrism{Float64}(width = 0.4, height = 0.3, depth = 0.2, center = V3(1, 2, 3),
                                        euler_angles = V3(0.3, -0.2, 1.1))             => 0.4 * 0.3 * 0.2,
              Cylinder{Float64}(radius = 0.2, height = 0.5, center = V3(0, 1, 0),
                                euler_angles = V3(0.5, 0, 0))                          => π * 0.2^2 * 0.5)
    e_mass = e_com = e_vel = 0.0
    for (shape, V) in shapes
        pos, vel, mass, vol = generate_particles(shape, h, ρ, v0, ω)
        e_mass = max(e_mass, abs(sum(mass) / (ρ * V) - 1))
        e_com  = max(e_com, norm(sum(mass .* pos) / sum(mass) - shape.center) / h)
        e_vel  = max(e_vel, maximum(norm.(vel .- (Ref(v0) .+ cross.(Ref(ω), pos .- Ref(shape.center))))))
        @test all(==(h^3), vol) && all(==(ρ * h^3), mass)
    end
    @test e_mass < 0.03          # volume up to the staircase surface
    @test e_com < 1e-10          # symmetric lattice around the center
    @test e_vel < 1e-12          # rigid motion v0 + ω × (x - c)
end

@testset "build_mpm_model: initial grid state ($(nameof(G)))" for G in (DenseGrid, SparseGrid)
    # Exact initial P2G (APIC, affine C = [ω]×) followed by the half kick dt0/2 * M g:
    #   M_grid = Σ m_p
    #   P_grid = Σ m_p v_p + dt0/2 M g
    #   L_grid = Σ m_p x_p × v_p + M ω Δx²/2  (APIC spin: Σ w r × (ω × r) = ω Δx²/2)
    #            + dt0/2 M x_cm × g
    g, v0, ω = V3(0, -9.81, 0), V3(1, 0, 0.5), V3(0.5, -1, 2)
    model = single_block(; grid_type = G, v0, ω, external_force = ConstantGravity(g))
    p     = model.particle_sets[1].particles
    x     = V3.(collect(p.pos))
    v     = Ref(v0) .+ cross.(Ref(ω), x)
    M, dx = sum(p.mass), 1 / model.grid.inv_dx
    dt0   = model.dt_previous
    x_cm  = sum(p.mass .* x) / M
    tot   = grid_totals(model.grid)

    @test tot.M ≈ M rtol = 1e-13
    @test tot.P ≈ sum(p.mass .* v) + dt0 / 2 * M * g rtol = 1e-12
    @test tot.L ≈ sum(p.mass .* cross.(x, v)) + M * ω * dx^2 / 2 + dt0 / 2 * M * cross(x_cm, g) rtol = 1e-12

    # first step: CFL time step from c + |v| of the particles
    c = model.particle_sets[1].material.c
    @test dt0 ≈ model.cfl_number * dx / maximum(vp -> c + norm(vp), v) rtol = 1e-12
    @test single_block(; grid_type = G, t_max = 1e-6).dt_previous == 1e-6      # clipped to t_max
end

@testset "build_mpm_model: domain options" begin
    sparse(; kw...) = colliding_blocks(; grid_type = SparseGrid, kw...).grid.domain
    @test sparse() isa OpenDomain                                         # open boundaries
    @test sparse(boundary_condition = NoSlipBoundary()) isa BoxDomain     # walls need a box
    @test sparse(domain = FittedBox()) isa BoxDomain
    @test_throws ErrorException sparse(domain = OpenDomain(), boundary_condition = NoSlipBoundary())
end


# ---------------------------------------------------------------------------- #
#                               Time integration                               #
# ---------------------------------------------------------------------------- #
# x^{n+1} = x^n + dt_n v^{n+1/2},  v^{n+3/2} = v^{n+1/2} + (dt_n + dt_{n+1})/2 a^{n+1}
# with v^{1/2} = v^0 + dt_0/2 a^0: exact for constant a, also with variable dt.
@testset "Free fall is exact ($(nameof(G)))" for G in (DenseGrid, SparseGrid)
    g, v0 = V3(0, -9.81, 0), V3(0.5, 0, 0)
    model = single_block(; grid_type = G, v0, external_force = ConstantGravity(g))
    x0    = center_of_mass(model)
    dts   = [step!(model) for _ in 1:20]
    t     = model.t
    disp  = v0 * t + g * t^2 / 2

    @test t ≈ sum(dts) rtol = 1e-14
    @test maximum(dts) > minimum(dts)                                   # dt really varies (|v| in CFL)
    @test norm(center_of_mass(model) - x0 - disp) < 1e-12 * norm(disp)
    @test norm(g) * dts[1] * t / 2 > 1e-3 * norm(disp)                  # a missing half kick would show
    # uniform acceleration: no deformation at all
    @test maximum(F -> norm(F - I3), model.particle_sets[1].particles.F) < 1e-12
end

@testset "Last step ends exactly at t_max" begin
    model = single_block(; v0 = V3(0.5, 0, 0), t_max = 1e-2)
    n = 0
    while model.t < model.t_max && n < 1000
        step!(model)
        n += 1
    end
    @test n > 1
    @test model.t ≈ model.t_max rtol = 1e-12
    @test model.dt_previous ≈ 0 atol = 1e-12
end


# ---------------------------------------------------------------------------- #
#                 Conservation laws and grid type equivalence                  #
# ---------------------------------------------------------------------------- #
"Momentum of particle set i, with the particle velocities interpolated from the grid (G2P)."
function set_momentum(model, i)
    ps   = model.particle_sets[i]
    v, _ = S.extract_velocities(model.grid, ps, model.shapefunction)
    return sum(ps.particles.mass .* V3.(v))
end

@testset "Conservation, DenseGrid ≡ SparseGrid: $(nameof(typeof(tr)))(heating=$(S.transfer_heating(tr)))" for
        tr in (APIC(), PolyPIC(heating = true))

    models = (dense = colliding_blocks(; transfer = tr),
              open  = colliding_blocks(; transfer = tr, grid_type = SparseGrid),
              box   = colliding_blocks(; transfer = tr, grid_type = SparseGrid, domain = FittedBox()))
    tot0   = map(m -> grid_totals(m.grid), models)
    P_a0   = set_momentum(models.dense, 1)
    P_ref  = sum(ps -> sum(ps.particles.mass), models.dense.particle_sets) * 1.0    # M × 1 m/s
    dev    = map(_ -> (M = 0.0, P = 0.0, L = 0.0), models)
    dt_err = 0.0
    for _ in 1:60
        dts = map(step!, models)
        dt_err = max(dt_err, abs(dts.open - dts.dense) / dts.dense, abs(dts.box - dts.dense) / dts.dense)
        dev = map(models, tot0, dev) do m, t0, d
            t = grid_totals(m.grid)
            (M = max(d.M, abs(t.M - t0.M) / t0.M), P = max(d.P, norm(t.P - t0.P) / P_ref),
             L = max(d.L, norm(t.L - t0.L) / norm(t0.L)))
        end
    end

    @testset "$name" for name in keys(models)
        @test dev[name].M < 1e-13
        @test dev[name].P < 1e-12
        @test dev[name].L < 1e-12          # APIC and PolyPIC conserve angular momentum exactly
    end

    # the bodies really collided: momentum was exchanged between them
    @test norm(set_momentum(models.dense, 1) - P_a0) > 0.2 * norm(P_a0)

    # all three grids follow the same trajectory (only rounding differences)
    x_dense = all_positions(models.dense)
    @test dt_err < 1e-12
    @test maximum(norm.(all_positions(models.open) .- x_dense)) < 1e-12
    @test maximum(norm.(all_positions(models.box)  .- x_dense)) < 1e-12
    @test S.missed_count(models.open.grid.map_read) == 0 && S.missed_count(models.box.grid.map_read) == 0
end


# ---------------------------------------------------------------------------- #
#                                    Energy                                    #
# ---------------------------------------------------------------------------- #
@testset "Elastic oscillation: kinetic ⇄ strain energy" begin
    # A free cube gets the uniform dilatation v = a x. The internal forces must turn
    # the kinetic energy (almost) completely into strain energy and back, without
    # creating or losing energy (sign or factor errors in σ -> force break this),
    # and with the stiffness of the material (quarter period).
    model = single_block(; dx = 0.05, cfl = 0.1, transfer = PolyPIC())
    set_grid_velocity!(model.grid, x -> 3.0 * x)
    E0 = energy_budget(model).total
    t, K, W = Float64[], Float64[], Float64[]
    while model.t < 0.0118                        # ≈ 1.4 periods
        step!(model)
        e = energy_budget(model)
        push!(t, model.t); push!(K, e.kinetic / E0); push!(W, e.elastic / E0)
    end
    @test minimum(K) < 0.15 && maximum(W) > 0.85
    @test 0.95 < minimum(K .+ W) && maximum(K .+ W) < 1.03
    @test 0.0039 < t[argmin(K)] < 0.0047          # quarter period of the breathing mode
end

@testset "Heat bookkeeping ($(nameof(G)), $scheme)" for G in (DenseGrid, SparseGrid), scheme in (APIC, PolyPIC)
    # pressureless dust: no internal forces, the collision is perfectly inelastic
    cold = run!(dust_collision(; grid_type = G, transfer = scheme()), 80)
    hot  = dust_collision(; grid_type = G, transfer = scheme(heating = true))
    E0   = energy_budget(hot).total
    err  = maximum(1:80) do _
        step!(hot)
        abs(energy_budget(hot).total - E0) / E0
    end
    E_cold, E_hot = energy_budget(cold), energy_budget(hot)

    @test E_cold.internal == 0 && (E0 - E_cold.total) / E0 > 0.1        # without heating: kinetic energy lost
    @test err < 1e-12                                                   # with heating: K + U + pending conserved
    @test E_hot.internal > 0.1 * E0
    e_ref = E0 / sum(ps -> sum(ps.particles.mass), hot.particle_sets)                  # J/kg
    @test minimum(ps -> minimum(st -> st.e, ps.particles.mat_state), hot.particle_sets) > -1e-12 * e_ref   # heat ≥ 0
    @test E_hot.kinetic ≈ E_cold.kinetic rtol = 1e-10                   # same dynamics (dust: e does not feed back)
end

@testset "PolyPIC dissipates less than APIC" begin
    lost(tr) = (m = dust_collision(; transfer = tr); E0 = energy_budget(m).total; run!(m, 40);
                E0 - energy_budget(m).total)
    @test lost(PolyPIC()) < lost(APIC())
end

@testset "Aluminium impact: energy budget" begin
    budget(tr) = begin
        m  = al_impact(; transfer = tr)
        E0 = energy_budget(m).total
        run!(m, 60)
        E  = energy_budget(m)
        (total = E.total / E0, internal = E.internal / E0, elastic = E.elastic / E0)
    end
    cold, apic, poly = budget(APIC()), budget(APIC(heating = true)), budget(PolyPIC(heating = true))

    # the impact converts kinetic energy into strain energy and heat (shock, volume work)
    @test all(r -> r.internal > 0.05 && r.elastic > 0.2, (cold, apic, poly))
    @test cold.total < 0.95                                    # APIC without heating loses energy
    @test abs(apic.total - 1) < 0.02 && abs(poly.total - 1) < 0.03
    @test apic.internal > cold.internal

    # With heating the transfer is exact, the rest is the O(Δt) error of the time
    # integration at the contact: it halves with Δt (same physical time)
    dE(cfl) = begin
        m  = al_impact(; transfer = PolyPIC(heating = true), cfl)
        E0 = energy_budget(m).total
        run!(m, round(Int, 31 * 0.4 / cfl))
        energy_budget(m).total / E0 - 1
    end
    d4, d2 = dE(0.4), dE(0.2)
    @test abs(d4) > 0.005 && 1.6 < d4 / d2 < 2.6
end

@testset "Heating is Galilean invariant (Float32, $(nameof(G)))" for G in (DenseGrid, SparseGrid)
    # rigid translation at 1000 m/s: physically no heat. The heat terms are evaluated
    # relative to reference velocities, so ½|v|² = 5.4e5 J/kg never has to cancel.
    T = Float32
    e_max = maximum((APIC(heating = true), PolyPIC(heating = true))) do tr
        body  = Body(RectangularPrism{T}(width = T(2e-3), height = T(2e-3), depth = T(2e-3)),
                     SVector{3, T}(1000, 300, 0), zero(SVector{3, T}), Aluminum(T))
        setup = SimulationSetup(dx = T(2.5e-4), t_max = T(1), padding = 4, buffer_width = SVector(8, 8, 8),
                                dt_max = T(1e-6), grid_type = G, transfer = tr)
        model = run!(build_mpm_model((body,), setup), 40)
        maximum(st -> abs(st.e), model.particle_sets[1].particles.mat_state)
    end
    @test e_max < 1f-2                                         # J/kg
end

@testset "Multithreaded runs only differ by rounding" begin
    # The atomic adds come in a different order in every run. A discontinuous branch
    # in the update would amplify that rounding noise within a few steps.
    if Threads.nthreads() == 1
        # no @test_skip: it would show up as an unexplained "Broken" in the summary
        @info "SKIPPED \"Multithreaded runs only differ by rounding\": start julia with -t 2 or more"
    else
        d = maximum((APIC(), PolyPIC())) do tr
            m1, m2 = al_impact(; transfer = tr), al_impact(; transfer = tr)
            run!(m1, 30); run!(m2, 30)
            maximum(norm.(all_positions(m1) .- all_positions(m2)))
        end
        @test d < 1e-10 * 2.5e-4
    end
end
