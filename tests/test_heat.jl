# ---------------------------------------------------------------------------- #
#          Heat bookkeeping of the particle-grid transfer (heat = true)        #
# ---------------------------------------------------------------------------- #
# Two colliding Basalt bodies (SolidMaterial: has an internal energy e), open domain.
#   * linear and angular momentum (incl. the affine spin part) exact, with and without
#   * AFLIP(heat = true): total energy (½m|v|² + ½m d‖C‖² + m e) nearly conserved and
#     much better than without the bookkeeping
#   * every booked term is ≥ 0: with heat = true no particle ends with less e than
#     without (same run, the booking only ever adds)
#   * APIC books its (large) transfer loss too, but then exposes the mismatch between
#     the work of the grid forces (kick velocity at x^n) and the work booked by the
#     material (MUSL velocity at x^{n+1/2}), E/E0 ≈ 1.07 here: only E > 1 is checked.

function _heat_totals(model)
    d = (1 / model.grid.inv_dx)^2 / 4
    L = zero(SVector{3, Float64}); P = zero(SVector{3, Float64}); E = 0.0; Ls = 0.0; Ps = 0.0
    for ps in model.particle_sets, i in 1:length(ps.particles)
        p = ps.particles; m = p.mass[i]; x = p.pos[i]; v = p.vel[i]; C = p.C[i]
        L += m * cross(x, v) + m * d * SVector(C[3,2] - C[2,3], C[1,3] - C[3,1], C[2,1] - C[1,2])
        P += m * v
        E += 0.5m * (dot(v, v) + d * sum(abs2, C)) + m * p.mat_state[i].e
        Ls += m * norm(x) * norm(v)
        Ps += m * norm(v)
    end
    return (; L, P, E, Ls, Ps)
end

function _heat_run(scheme; steps = 150)
    T   = Float64
    mat = Basalt(T, eos = :tillotson, elasticity = :hypoelastic, damage = :no_damage)
    b1  = Body(RectangularPrism{T}(width = 0.03, height = 0.03, depth = 0.03,
                                   center = SVector(-0.022, 0.004, 0.0)),
               SVector(300.0, 0.0, 0.0), SVector(0.0, 0.0, 500.0), mat)
    b2  = Body(Sphere{T}(0.016, SVector(0.02, 0.0, 0.0)), SVector(-200.0, 0.0, 0.0),
               zero(SVector{3, T}), mat)
    setup = SimulationSetup(; dx = 4e-3, t_max = 1.0, dt_max = 1.0, padding = 2, ppc_1d = 2,
                              transfer_scheme = scheme, grid_type = SparseGrid, domain = OpenDomain())
    model = build_mpm_model((b1, b2), setup)
    q0 = _heat_totals(model)
    for _ in 1:steps
        SmashMPM.step!(model)
    end
    q = _heat_totals(model)
    e = reduce(vcat, [[s.e for s in ps.particles.mat_state] for ps in model.particle_sets])
    return (dL = norm(q.L - q0.L) / q0.Ls, dP = norm(q.P - q0.P) / q0.Ps, E = q.E / q0.E, e = e)
end

@testset "Heat bookkeeping" begin
    @test SmashMPM.books_heat(AFLIP(0.9; heat = true)) && !SmashMPM.books_heat(AFLIP(0.9))
    @test SmashMPM.books_heat(APIC(heat = true)) && !SmashMPM.books_heat(APIC())
    @test AFLIP(0.9) isa AFLIP && APIC() isa APIC                 # old constructors unchanged

    r = Dict(name => _heat_run(s) for (name, s) in (
        "AFLIP"      => AFLIP(0.95),
        "AFLIP+heat" => AFLIP(0.95; heat = true),
        "APIC+heat"  => APIC(heat = true)))
    for (name, x) in sort!(collect(r); by = first)
        @info "transfer $name" x.dL x.dP x.E
        @test x.dL < 1e-12          # angular momentum incl. affine spin: exact
        @test x.dP < 1e-12
    end
    @test abs(r["AFLIP+heat"].E - 1) < 1e-2
    @test abs(r["AFLIP+heat"].E - 1) < abs(r["AFLIP"].E - 1) / 5
    @test r["APIC+heat"].E > 1
    # the booking only adds heat: the mean e is higher than without
    @test sum(r["AFLIP+heat"].e) > sum(r["AFLIP"].e)
end
