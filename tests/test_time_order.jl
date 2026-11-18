using Printf

# ---------------------------------------------------------------------------- #
#             2nd order in time of the 2-pass KDK / MUSL G2P2G                 #
# ---------------------------------------------------------------------------- #
# Self-convergence in Δt at a fixed grid: the same model is run with Δt = h, h/2,
# h/4, h/8 up to the same time, and the differences of the particle positions
# and deformation gradients between successive refinements are compared,
#   e_k = ‖q_{h/2^k} - q_{h/2^(k+1)}‖,   order ≈ log2(e_k / e_{k+1}).
# Positions and F live on full time levels (velocities are half steps and are
# not compared). Rotating NeoHookean cube under gravity, open boundaries:
# elastic waves + rigid rotation, no contact.
#
# Cases:
#   * constant Δt
#   * variable Δt (smooth function of t): checks the time-centered kick
#     Δt̄_n = (Δt_{n-1} + Δt_n) / 2 and the external force in the force field.
# Pure FLIP (α = 1) is the case that converges in Δt at a fixed grid: for α < 1
# the per-step PIC blend 1-α is a damping per step, not per time, so it has no
# Δt -> 0 limit at a fixed grid (neither has the APIC projection), and only the
# orders of AFLIP(1) are asserted. APIC is run as a smoke test of the variable Δt.

function _order_model(scheme; ext_force = ConstantGravity(SVector(0.0, 0.0, -9.81)))
    T     = Float64
    mat   = NeoHookean(ρ = 1000.0, E = 1e6, ν = 0.3)
    shape = RectangularPrism{T}(width = 0.6, height = 0.6, depth = 0.6)
    body  = Body(shape, SVector(0.3, 0.0, 0.0), SVector(0.0, 0.0, 2.0), mat)
    setup = SimulationSetup(; dx = 0.1, t_max = 1.0, cfl_number = 1.0, dt_max = 1e-3,
                              padding = 2, ppc_1d = 2, transfer_scheme = scheme,
                              external_force = ext_force)
    return build_mpm_model((body,), setup)
end

"""Run up to `t_end` with Δt(t) = h * g(t); the last step is clipped to land on t_end."""
function _order_run(scheme, h, t_end; g = t -> 1.0)
    model = _order_model(scheme)
    model.t_max = t_end
    while model.t < t_end * (1 - 1e-14)
        t0 = model.t
        model.dt_max = h * g(t0)
        dt = SmashMPM.step!(model)
        # the Courant limit must not interfere, Δt is prescribed
        @assert dt == min(h * g(t0), t_end - t0) "Courant limit active"
    end
    P = model.particle_sets[1].particles
    return (pos = collect(P.pos), F = collect(P.F), model = model)
end

_fmt(v, f) = join((Printf.format(Printf.Format(f), x) for x in v), "  ")
_order_report(name, r) = println(rpad(name, 24),
    "e_x = ", _fmt(r.e_x, "%.2e"), "   order x = ", _fmt(r.p_x, "%.2f"),
    "   |   order F = ", _fmt(r.p_F, "%.2f"))

_order_err(a, b) = sqrt(sum(norm(x - y)^2 for (x, y) in zip(a, b)) / length(a))

function _order_study(scheme; h = 8e-4, t_end = 0.04, levels = 4, g = t -> 1.0)
    runs = [_order_run(scheme, h / 2^k, t_end; g) for k in 0:levels-1]
    e_x  = [_order_err(runs[k].pos, runs[k+1].pos) for k in 1:levels-1]
    e_F  = [_order_err(runs[k].F,   runs[k+1].F)   for k in 1:levels-1]
    p_x  = [log2(e_x[k] / e_x[k+1]) for k in 1:levels-2]
    p_F  = [log2(e_F[k] / e_F[k+1]) for k in 1:levels-2]
    return (; e_x, e_F, p_x, p_F, runs)
end

@testset "2nd order in time (KDK, 2-pass MUSL)" begin
    @testset "kick_dt" begin
        @test SmashMPM.kick_dt(0.0, 1e-3) == 5e-4               # first step: half kick
        @test SmashMPM.kick_dt(1e-3, 3e-3) == 2e-3              # time-centered
    end

    @testset "AFLIP(1), constant Δt" begin
        r = _order_study(AFLIP(1.0))
        _order_report("AFLIP(1), constant Δt", r)
        @test all(>(1.8), r.p_x)
        @test all(>(1.8), r.p_F)
    end

    @testset "AFLIP(1), variable Δt" begin
        g(t) = 1 + 0.5 * sin(2π * t / 0.04)                     # Δt varies by a factor of 3
        r = _order_study(AFLIP(1.0); g)
        _order_report("AFLIP(1), variable Δt", r)
        @test all(>(1.8), r.p_x)
        @test all(>(1.8), r.p_F)
        @test r.runs[end].model.dt_prev > 0
    end

    @testset "APIC, variable Δt (smoke)" begin
        g(t) = 1 + 0.5 * sin(2π * t / 0.04)
        r = _order_run(APIC(), 4e-4, 0.04; g)
        @test r.model.t ≈ 0.04
        @test all(F -> all(isfinite, F), r.F)
    end
end

@testset "format_sim_time" begin
    f = SmashMPM.format_sim_time
    @test f(0.0)        == "0 s"
    @test f(1.234567e-9) == "1.235 ns"
    @test f(12.345678e-6) == "12.35 µs"
    @test f(999.96e-9)  == "1 µs"            # rounding moves to the next unit
    @test f(0.0035)     == "3.5 ms"
    @test f(42.1666)    == "42.17 s"
    @test f(450.0)      == "7.5 min"
    @test f(8100.0)     == "2.25 h"
    @test f(267840.0)   == "3.1 d"
    @test f(-2.5e-6)    == "-2.5 µs"
    @test f(8.1234e-5; digits=3) == "81.2 µs"
end
