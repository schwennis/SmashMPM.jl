# ---------------------------------------------------------------------------- #
#                               Material models                                #
# ---------------------------------------------------------------------------- #
# Physics checks on the level of one material point:
#   - stress is the derivative of the stored energy (no spurious energy sources)
#   - objectivity (rigid rotations produce no stress)
#   - EoS: analytic values, continuity, thermodynamic consistency of c and ∂p/∂e
#   - return mapping: consistency condition and closed-form solutions
#   - damage: explicit Weibull flaws (Benz & Asphaug), growth law, caps
#   - SolidMaterial: reference state, stress assembly, energy update (heat ≥ 0,
#     isentropic loops close, second order in Δt)

"Rotation matrix exp(skew(θ))."
function rotation(θ::SVector{3})
    K = @SMatrix [0 -θ[3] θ[2]; θ[3] 0 -θ[1]; -θ[2] θ[1] 0]
    return SMatrix{3, 3}(exp(Matrix(K)))
end

"Central finite difference of f along a direction: (f(x + h d) - f(x - h d)) / 2h."
fd(f, x, d, h) = (f(x + h * d) - f(x - h * d)) / 2h

kin(F, C; V0 = 1.0, m = 1.0, dt = 1e-7, dx = 1e-3) = S.Kinematics(M3(F), M3(C), V0, m, dt, dx)


# ---------------------------------------------------------------------------- #
#                           Simple hyperelastic laws                           #
# ---------------------------------------------------------------------------- #
@testset "LinearElastic / NeoHookean" begin
    E, ν, ρ = 2e9, 0.3, 1200.0
    λ, μ    = E * ν / ((1 + ν) * (1 - 2ν)), E / (2 * (1 + ν))
    rng     = Xoshiro(3)
    stress(mat, F) = first(material_model(mat, NoMaterialState(), F, zero(M3), 1.0, 1.0, 1e-6, 1.0))

    for Mat in (LinearElastic, NeoHookean)
        mat = Mat(E = E, ν = ν, ρ = ρ)
        @testset "$Mat" begin
            # Lamé parameters, P-wave speed, both constructors agree, ρ is required
            @test mat.λ ≈ λ && mat.μ ≈ μ
            @test soundspeed(mat, NoMaterialState()) ≈ sqrt((λ + 2μ) / ρ)
            @test Mat(λ = λ, μ = μ, ρ = ρ).c ≈ mat.c
            @test_throws ErrorException Mat(E = E, ν = ν)
            @test iszero(stress(mat, I3))                           # reference state is stress free

            # stress power: dW/dt = J σ : L  for  Ḟ = L F  (NeoHookean, exact)
            #               ∂W/∂F = σ                   (LinearElastic, small strain law)
            W(F) = S.strain_energy(mat, NoMaterialState(), F)
            err = 0.0
            for _ in 1:20
                F = I3 + 0.15 * M3(randn(rng, 3, 3))
                det(F) > 0.5 || continue
                L = M3(randn(rng, 3, 3))
                σ = stress(mat, F)
                if mat isa NeoHookean
                    lhs, rhs = fd(G -> W(G), F, L * F, 1e-6), det(F) * dot(σ, L)
                else
                    lhs, rhs = fd(G -> W(G), F, L, 1e-6), dot(σ, L)
                end
                err = max(err, abs(lhs - rhs) / (μ * norm(L)))
            end
            @test err < 1e-6
        end
    end

    # Objectivity of the finite-strain law: σ(R F) = R σ(F) Rᵀ, σ(R) = 0
    neo = NeoHookean(E = E, ν = ν, ρ = ρ)
    err = 0.0
    for _ in 1:20
        R = rotation(SVector{3}(2 .* randn(rng, 3)))
        F = I3 + 0.2 * M3(randn(rng, 3, 3))
        det(F) > 0.5 || continue
        err = max(err, relerr(stress(neo, R * F), R * stress(neo, F) * R'),
                       norm(stress(neo, R)) / μ)
    end
    @test err < 1e-10
end


# ---------------------------------------------------------------------------- #
#                             Equations of state                               #
# ---------------------------------------------------------------------------- #
@testset "Murnaghan EoS" begin
    ρ0, K0, n, η_lim = 2860.0, 26.7e9, 5.5, 0.9
    eos = MurnaghanEOS{Float64}(ρ0, K0, n, η_lim)
    st0 = init_state(eos)
    p(ρ, e = 0.0) = update_eos(eos, st0, ρ, e)[1].p
    c(ρ)          = update_eos(eos, st0, ρ, 0.0)[2]

    ηs = range(0.92, 1.4; length = 40)
    @test p(ρ0) == 0 && S.reference_soundspeed(eos) ≈ sqrt(K0 / ρ0) ≈ c(ρ0)
    @test maximum(η -> abs(p(η * ρ0) - K0 / n * (η^n - 1)), ηs) < 1e-6 * K0
    @test maximum(η -> abs(c(η * ρ0)^2 - fd(p, η * ρ0, 1.0, 1e-3)) / c(η * ρ0)^2, ηs) < 1e-6   # c² = dp/dρ
    @test p(1.1ρ0, 1e7) == p(1.1ρ0)                         # barotropic: p independent of e
    @test p(0.85ρ0) == 0 && c(0.85ρ0) ≈ c(η_lim * ρ0)        # tension cutoff
end

@testset "Tillotson EoS ($name)" for (name, eos) in (("iron", Iron(Float64).eos), ("basalt", Basalt(Float64).eos))
    st0 = init_state(eos)
    p(ρ, e)    = update_eos(eos, st0, ρ, e)[1].p
    c(ρ, e)    = update_eos(eos, st0, ρ, e)[2]
    dpde(ρ, e) = S.update_eos_dpde(eos, st0, ρ, e)[3]
    ρ0, A, B   = eos.ρ0, eos.A, eos.B
    Eiv, Ecv   = eos.Eiv, eos.Ecv

    # reference state and cold compression p = A μ + B μ²
    @test p(ρ0, 0.0) == 0
    @test c(ρ0, 0.0) ≈ sqrt(A / ρ0) ≈ S.reference_soundspeed(eos)
    @test maximum(μ -> abs(p((1 + μ) * ρ0, 0.0) - (A * μ + B * μ^2)) / A, 0:0.02:0.4) < 1e-12

    # Thermodynamic consistency in every regime:
    #   ∂p/∂e from update_eos_dpde = finite difference,   c² = ∂p/∂ρ + p/ρ² ∂p/∂e
    points = [(1.1ρ0, 1e6),                  # condensed
              (0.95ρ0, 0.5Eiv),              # cold expansion (condensed form)
              (0.9ρ0, 0.5 * (Eiv + Ecv)),    # hybrid
              (0.8ρ0, 2Ecv)]                 # hot expansion
    eos.ρiv > 0 && push!(points, (0.6ρ0, 0.5Eiv))  # vapor branch (ρ < ρiv)
    e_dpde = e_c = 0.0
    for (ρ, e) in points
        dpde_fd = fd(x -> p(ρ, x), e, 1.0, 1e-6 * e)
        dpdρ_fd = fd(x -> p(x, e), ρ, 1.0, 1e-7 * ρ)
        e_dpde  = max(e_dpde, abs(dpde(ρ, e) - dpde_fd) / abs(dpde_fd))
        e_c     = max(e_c, abs(c(ρ, e)^2 - (dpdρ_fd + p(ρ, e) / ρ^2 * dpde_fd)) / c(ρ, e)^2)
    end
    @test e_dpde < 1e-6
    @test e_c < 1e-5

    # continuity across ρ = ρ0 and across the hybrid region boundaries Eiv, Ecv
    jump(f, x, h) = abs(f(x + h) - f(x - h))
    @test maximum(e -> jump(ρ -> p(ρ, e), ρ0, 1e-9ρ0), (0.5Eiv, 0.5 * (Eiv + Ecv), 2Ecv)) < 1e-6 * A
    @test maximum(e -> jump(x -> p(0.9ρ0, x), e, 1e-6), (Eiv, Ecv)) < 1e-9 * A

    # the linearized energy correction p + ∂p/∂e Δe is second order in Δe
    ρ, e = 0.9ρ0, 0.5 * (Eiv + Ecv)
    st, _, k = S.update_eos_dpde(eos, st0, ρ, e)
    err(Δe) = abs(S.correct_eos_energy(eos, st, ρ, e + Δe, k, Δe).p - p(ρ, e + Δe))
    @test err(1e3) > 0 && 3 < err(2e3) / err(1e3) < 5
end


# ---------------------------------------------------------------------------- #
#                                  Elasticity                                  #
# ---------------------------------------------------------------------------- #
@testset "Elasticity" begin
    μ, dt = 30e9, 1e-6
    shear = @SMatrix [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    spin  = @SMatrix [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]

    @testset "HypoElasticity (Jaumann rate)" begin
        el = HypoElasticity{Float64}(μ)
        st = init_state(el)
        γ̇  = 10.0
        s, _, G = S.elastic_trial(el, st, kin(I3, γ̇ * shear; dt))
        @test G == μ && s ≈ 2μ * dt * γ̇ * shear              # Δs = 2μ dev(D) Δt
        s, _, _ = S.elastic_trial(el, st, kin(I3, 5.0 * I3; dt))
        @test iszero(s)                                       # pure dilatation: no deviator

        # Rigid spin of a prestressed point: the stress is rotated, not changed
        s0 = 1e8 * M3(0.0, 1.0, 0.5, 1.0, -2.0, 0.0, 0.5, 0.0, 2.0)
        ω, n = 1e3, 1000                                      # total angle ω n dt = 1 rad
        st = S.HypoElasticState(s0)
        for _ in 1:n
            s, st, _ = S.elastic_trial(el, st, kin(I3, ω * spin; dt))
            st = S.consistent_elastic_state(el, st, s, kin(I3, ω * spin; dt))
        end
        R = rotation(SVector(0.0, 0.0, ω * n * dt))
        @test relerr(st.s, R * s0 * R') < 2e-3                # first order in ω dt
    end

    @testset "HyperElasticity (isochoric neo-Hooke)" begin
        el = HyperElasticity{Float64}(μ)
        st = init_state(el)
        γ  = 0.05
        F  = I3 + γ * @SMatrix [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]   # simple shear, det F = 1
        s, st_tr, _ = S.elastic_trial(el, st, kin(F, (F - I3) / dt; dt))
        b = F * F'
        @test s ≈ μ * (b - tr(b) / 3 * I3)                   # s = μ dev(b̄)
        s_vol, _, _ = S.elastic_trial(el, st, kin(1.05 * I3, (0.05 / dt) * I3; dt))
        @test norm(s_vol) < 1e-9 * μ                         # volumetric stretch: no deviator

        # after a plastic correction the elastic state keeps det b̄ₑ = 1, dev(b̄ₑ) = J s / μ
        s_pl  = 0.6 * s
        st_pl = S.consistent_elastic_state(el, st_tr, s_pl, kin(F, zero(M3); dt))
        @test det(st_pl.bbar_e) ≈ 1 atol = 1e-12
        @test st_pl.bbar_e - tr(st_pl.bbar_e) / 3 * I3 ≈ s_pl / μ
    end
end


# ---------------------------------------------------------------------------- #
#                                  Plasticity                                  #
# ---------------------------------------------------------------------------- #
@testset "J2 plasticity (radial return)" begin
    μ, σ_Y = 105e9, 2.5e8
    R      = sqrt(2 / 3) * σ_Y                         # radius of the yield surface ‖s‖
    dir    = M3(3.0, 1.0, 0.5, 1.0, -1.0, 2.0, 0.5, 2.0, -2.0)
    dir    = dir / norm(dir)
    k      = kin(I3, zero(M3))
    rmap(pl, st, s) = S.return_map(pl, st, s, μ, nothing, k)

    # inside the yield surface: nothing changes
    pl = J2Plasticity(σ_y0 = σ_Y, H = 1e9)
    st = init_state(pl)
    @test all(x -> rmap(pl, st, x * R * dir) == (x * R * dir, st), (0.0, 0.5, 0.999))

    # outside: back onto the (hardened) surface along the trial direction, closed form for α
    e_cons = e_dir = e_alpha = 0.0
    for H in (0.0, 1e9, 5e10), x in (1.01, 2.0, 5.0)
        pl = J2Plasticity(σ_y0 = σ_Y, H = H)
        s_tr = x * R * dir
        s, st_new = rmap(pl, init_state(pl), s_tr)
        α = st_new.eps_p
        e_cons  = max(e_cons, abs(norm(s) - sqrt(2 / 3) * (σ_Y + H * α)) / R)      # f(s, α) = 0
        e_dir   = max(e_dir, norm(s / norm(s) - dir))
        e_alpha = max(e_alpha, abs(α - sqrt(2 / 3) * (norm(s_tr) - R) / (2μ + 2H / 3)) / α)
    end
    @test e_cons < 1e-12 && e_dir < 1e-12 && e_alpha < 1e-12

    # hardening is remembered: elastic once α has raised the yield stress enough
    pl = J2Plasticity(σ_y0 = σ_Y, H = 1e10)
    st_hard = S.J2PlasticState(σ_Y / 1e10)                  # σ_y = 2 σ_Y
    @test rmap(pl, st_hard, 1.5R * dir) == (1.5R * dir, st_hard)

    # proportional loading is path independent (one step = two steps), and the
    # returned state is converged (returning it again changes nothing)
    pl = J2Plasticity(σ_y0 = σ_Y, H = 2e10)
    s_1, st_1 = rmap(pl, init_state(pl), 3.5R * dir)
    s_a, st_a = rmap(pl, init_state(pl), 2.0R * dir)
    s_2, st_2 = rmap(pl, st_a, s_a + 1.5R * dir)
    @test relerr(s_2, s_1) < 1e-12 && abs(st_2.eps_p - st_1.eps_p) < 1e-12 * st_1.eps_p
    @test relerr(first(rmap(pl, st_1, s_1)), s_1) < 1e-10

    # mixed precision: Float64 parameters, Float32 kinematics -> computed in Float32,
    # the state keeps its own type (the particle arrays store it)
    k32  = S.Kinematics(one(SMatrix{3, 3, Float32, 9}), zero(SMatrix{3, 3, Float32, 9}), 1f0, 1f0, 1f-7, 1f-3)
    s32  = SMatrix{3, 3, Float32, 9}(2R * dir)
    s, st = @inferred S.return_map(J2Plasticity(σ_y0 = σ_Y, H = 1e9), S.J2PlasticState(0.0), s32, Float32(μ), nothing, k32)
    @test eltype(s) === Float32 && st isa S.J2PlasticState{Float64} && st.eps_p > 0
end


# ---------------------------------------------------------------------------- #
#                         Damage and artificial viscosity                      #
# ---------------------------------------------------------------------------- #
@testset "Grady-Kipp damage" begin
    dm  = GradyKippDamage{Float64}(E = 53.1e9, m = 8.5, k = 3.0e39)   # basalt
    V   = 4 / 3 * π * 0.03^3
    n_p = 1000
    k(dt) = kin(I3, zero(M3); V0 = V / n_p, m = 2700.0 * V / n_p, dt = dt)
    c = 5000.0

    # Explicit Weibull flaws (Benz & Asphaug 1995): flaw j has ε_j = (j / kV)^(1/m) and
    # belongs to a random particle, until every particle has at least one flaw
    dm_b, states = S.init_states(dm, Xoshiro(1), n_p, V)
    flaws  = Float64.(dm_b.flaws)
    N      = length(flaws)
    kV     = dm.k * V
    own(st) = flaws[st.first_flaw:st.first_flaw + st.n_flaws - 1]
    @test all(st -> st.D == 0 && st.n_flaws >= 1, states)
    @test states[1].first_flaw == 1 && sum(st -> st.n_flaws, states) == N           # CSR: no gaps,
    @test all(i -> states[i + 1].first_flaw == states[i].first_flaw + states[i].n_flaws, 1:n_p - 1)  # no overlap
    @test all(st -> issorted(own(st)), states)
    @test sort(flaws) ≈ ((1:N) ./ kV) .^ (1 / dm.m) rtol = 1e-6                     # Weibull thresholds
    @test minimum(flaws) ≈ kV^(-1 / dm.m) rtol = 1e-6                               # weakest flaw of the body
    @test 0.5 * n_p * log(n_p) < N < 3 * n_p * log(n_p)                              # coupon collector ≈ n_p ln n_p
    @test S.init_states(dm, Xoshiro(1), n_p, V)[1].flaws == dm_b.flaws               # reproducible
    @test_throws ErrorException init_state(dm, Xoshiro(1), n_p, V)                   # no per-particle sampling

    # Growth law with three known flaws
    dmf = GradyKippDamage{Float64}(E = 53.1e9, m = 8.5, k = 3.0e39, flaws = Float32[1e-4, 2e-4, 4e-4])
    st  = S.GradyKippState(0.0, Int32(1), Int32(3))
    upd(st, p, dt) = S.update_damage(dmf, st, zero(M3), S.NullState(), (p = p, T = 0.0), k(dt), c)
    @test upd(st, 1e9, 1e-7) == st                                  # compression: no growth
    @test upd(st, -0.9e-4 * dmf.E, 1e-7) == st                      # below the weakest flaw
    @test upd(S.init_state(dmf), -1e-2 * dmf.E, 1.0).D == 0         # flawless particle

    # tension with ε = 2.5e-4: two active flaws, dD^⅓/dt = n_act c_g / R_s, D <= n_act / n_f
    p_t = -2.5e-4 * dmf.E
    R_s = cbrt(3 * V / n_p / (4π))
    @test cbrt(upd(st, p_t, 1e-11).D) ≈ 1e-11 * 2 * 0.4c / R_s
    st_max = upd(st, p_t, 1.0)
    @test st_max.D ≈ 2 / 3
    @test upd(st_max, -0.5e-4 * dmf.E, 1.0) == st_max                # damage never heals
    @test upd(st_max, -1e-3 * dmf.E, 1.0).D == 1                     # all flaws active: destroyed

    st_d = S.GradyKippState(0.25, Int32(1), Int32(3))
    @test S.degradation(dm, st_d) == 0.75
    @test S.degrade_pressure(dm, st_d, -4.0) == -3.0 && S.degrade_pressure(dm, st_d, 4.0) == 4.0

    # analytic largest eigenvalue = LAPACK
    rng = Xoshiro(5)
    @test maximum(1:100) do _
        A = M3(randn(rng, 3, 3)); A = A + A'
        abs(S.max_principal_stress(A) - maximum(eigvals(Symmetric(Matrix(A))))) / norm(A)
    end < 1e-10
end

@testset "Bulk viscosity" begin
    av = BulkViscosity(c_quad = 1.5, c_lin = 0.06)
    c, dx, ρ = 3000.0, 0.01, 1000.0
    @test S.viscous_pressure(av, kin(I3, 0.1 * I3; m = ρ, V0 = 1.0, dx = dx), c) == 0     # expansion: off
    divv = -10.0
    q = S.viscous_pressure(av, kin(I3, divv / 3 * I3; m = ρ, V0 = 1.0, dx = dx), c)
    @test q ≈ ρ * dx * (1.5 * dx * divv^2 - 0.06 * c * divv) && q > 0
end


# ---------------------------------------------------------------------------- #
#                           SolidMaterial pipeline                             #
# ---------------------------------------------------------------------------- #
"""
Drive one material point with a prescribed velocity gradient C(t), exactly like
the G2P2G kernel (trial F -> material model -> regularized F is stored).
Returns the final state, F, the external work Σ ½(τ_n + τ_{n+1}) : D dt V0 and
the last Cauchy stress.
"""
function drive(mat, st, Cfun, t_end, n; V0 = 1e-6, dx = 1e-3)
    dt, m, F = t_end / n, mat.ρ * V0, I3
    τ_old, work, σ = zero(M3), 0.0, zero(M3)
    for i in 1:n
        C  = Cfun((i - 0.5) * dt)
        Ft = (I3 + C * dt) * F
        σ, st_new = material_model(mat, st, Ft, C, V0, m, dt, dx)
        F  = S.regularize_deformation(mat, st, Ft)
        work += 0.5 * dt * dot(τ_old + det(F) * σ, (C + C') / 2) * V0
        τ_old, st = det(F) * σ, st_new
    end
    return st, F, work, σ
end

von_mises(σ) = sqrt(1.5) * norm(σ - tr(σ) / 3 * I3)

with_energy(st, e) = SolidMaterialState(c = st.c, e = e, eos_state = st.eos_state, elastic_state = st.elastic_state,
                                        plastic_state = st.plastic_state, damage_state = st.damage_state)
with_damage(st, D) = SolidMaterialState(c = st.c, e = st.e, eos_state = st.eos_state, elastic_state = st.elastic_state,
                                        plastic_state = st.plastic_state, damage_state = S.GradyKippState(D, Int32(1), Int32(0)))

@testset "SolidMaterial: reference state, stress assembly" begin
    V0, dx, dt = 1e-6, 1e-3, 1e-7

    # every configuration: the undeformed point at rest stays stress free and cold
    worst = 0.0
    for (_, mat) in material_zoo(Float64)
        mat isa SolidMaterial || continue
        st0 = initial_material_state(mat)
        σ, st1 = material_model(mat, st0, I3, zero(M3), V0, mat.ρ * V0, dt, dx)
        worst = max(worst, norm(σ) / 1e9, abs(st1.e), abs(st1.c - st0.c) / st0.c)
    end
    @test worst < 1e-12

    # hydrostatic compression (Murnaghan, no strength): σ = -p(ρ0 / J) I
    mat = SolidMaterial(eos = MurnaghanEOS{Float64}(2790.0, 26.7e9, 5.5, 0.9),
                        elasticity = HypoElasticity{Float64}(2.31e10), ρ = 2790.0)
    J   = 0.97
    σ, st = material_model(mat, initial_material_state(mat), cbrt(J) * I3, ((cbrt(J) - 1) / dt) * I3,
                           V0, mat.ρ * V0, dt, dx)
    p   = update_eos(mat.eos, init_state(mat.eos), mat.ρ / J, 0.0)[1].p
    @test σ ≈ -p * I3 && st.eos_state.p ≈ p
    @test st.e ≈ 0.5 * p * (1 - J) / mat.ρ rtol = 1e-10      # trapezoidal volume work ½(p_n + p) ΔV

    # σ = s - (p + q) I with deviator, EoS and viscosity
    mat = Basalt(Float64; eos = :tillotson, elasticity = :hypoelastic, damage = :no_damage,
                 viscosity = BulkViscosity())
    st0 = initial_material_state(mat)
    C   = @SMatrix [-100.0 10.0 0.0; 10.0 -100.0 0.0; 0.0 0.0 -100.0]
    F   = I3 + C * dt
    σ, st1 = material_model(mat, st0, F, C, V0, mat.ρ * V0, dt, dx)
    q   = S.viscous_pressure(mat.viscosity, S.Kinematics(F, C, V0, mat.ρ * V0, dt, dx), st0.c)
    @test σ ≈ st1.elastic_state.s - (st1.eos_state.p + q) * I3
    @test q > 0 && st1.e > 0                                  # compression heats

    # dt = 0 (used by the export): the stored state is returned unchanged
    _, st2 = material_model(mat, st1, F, zero(M3), V0, mat.ρ * V0, 0.0, dx)
    @test st2.e == st1.e && st2.elastic_state == st1.elastic_state

    # material configurations: every option builds in both precisions, typos fail early
    configs = Iterators.product((Float32, Float64), (:tillotson, :murnaghan), (:hypoelastic, :hyperelastic))
    @test all(configs) do (T, eos, el)
        Iron(T; eos, elasticity = el, damage = :grady_kipp) isa SolidMaterial{T} &&
            Basalt(T; eos, elasticity = el) isa SolidMaterial{T}
    end
    @test_throws ErrorException Iron(Float64; eos = :ideal_gas)
    @test_throws ErrorException Basalt(Float64; elasticity = :plastic)
end

@testset "SolidMaterial: energy update" begin
    V0, dx = 1e-6, 1e-3
    murnaghan = MurnaghanEOS{Float64}(2790.0, 26.7e9, 5.5, 0.9)

    # Isentropic compression loop J: 1 -> 0.9 -> 1 (Tillotson, no strength): e returns
    # to 0, and the trapezoidal update in p is second order (explicit p_n: first)
    mat = SolidMaterial(eos = Basalt(Float64).eos, elasticity = HypoElasticity{Float64}(0.0), ρ = 2790.0)
    function cycle(n)
        dt, st, J, e_max = 1e-5 / n, initial_material_state(mat), 1.0, 0.0
        for i in 1:n
            J_new = 1 - 0.1 * sin(π * i / n)^2
            C = ((cbrt(J_new / J) - 1) / dt) * I3
            _, st = material_model(mat, st, cbrt(J_new) * I3, C, V0, mat.ρ * V0, dt, dx)
            J, e_max = J_new, max(e_max, st.e)
        end
        return abs(st.e), e_max
    end
    (e1, e_max), (e2, _) = cycle(100), cycle(200)
    @test e1 < 1e-4 * e_max
    @test e1 / e2 > 3

    # Elastic shear cycle: (almost) no heat, for both elasticity models
    γ̇ = 50.0
    Cs(t) = γ̇ * sin(2π * t / 1e-4) * @SMatrix [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    W_max = 2.31e10 * (γ̇ * 1e-4 / π)^2 / 4 / 2790.0          # rough peak elastic energy per mass
    @test maximum((HyperElasticity{Float64}(2.31e10), HypoElasticity{Float64}(2.31e10))) do el
        mat = SolidMaterial(eos = murnaghan, elasticity = el, ρ = 2790.0)
        abs(drive(mat, initial_material_state(mat), Cs, 1e-4, 400)[1].e)
    end < 1e-3 * W_max

    # Plastic shear: external work = heat + stored elastic energy, heat > 0, and the
    # von Mises stress sits on the hardened yield surface
    σ_y, H = 1e8, 1e9
    mat = SolidMaterial(eos = murnaghan, elasticity = HyperElasticity{Float64}(2.31e10),
                        plasticity = J2Plasticity{Float64}(σ_y0 = σ_y, H = H), ρ = 2790.0)
    st, F, work = drive(mat, initial_material_state(mat), t -> 2e3 * @SMatrix([0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]), 1e-4, 1000)
    W_el = S.elastic_energy(mat.elasticity, st.elastic_state, det(F)) * V0
    @test st.e > 0
    @test work ≈ mat.ρ * V0 * st.e + W_el rtol = 1e-3
    τ = mat.elasticity.μ * (st.elastic_state.bbar_e - tr(st.elastic_state.bbar_e) / 3 * I3)
    @test sqrt(1.5) * norm(τ / det(F)) ≈ σ_y + H * st.plastic_state.eps_p rtol = 1e-6

    # No jump of the update at e = 0 (Tillotson clips e < 0): otherwise rounding noise
    # of the atomics is amplified and multithreaded runs diverge
    al  = Aluminum(Float64)
    st0 = initial_material_state(al)
    C   = -1e6 * I3
    stp(e) = material_model(al, with_energy(st0, e), I3 + 1e-8 * C, C, 1e-12, 2.7e-9, 1e-8, 2.5e-4)
    (σp, sp), (σm, sm) = stp(1e-20), stp(-1e-20)
    @test sp.eos_state.p > 1e9
    @test abs(sp.eos_state.p - sm.eos_state.p) < 1e-12 * sp.eos_state.p && abs(sp.e - sm.e) < 1e-9
end

@testset "SolidMaterial: damage and plasticity inside material_model" begin
    # Grady-Kipp (basalt, flaws of a body of n_p points): tension breaks the point and removes
    # the tensile stress, compression leaves it intact and identical to the undamaged material.
    # The point with the most flaws is driven (fastest growth, D = 1 well within 1 µs).
    n_p, V0 = 1000, 1e-6
    intact = Basalt(Float64; damage = :no_damage)
    gk, states_gk = initial_material_states(Basalt(Float64), Xoshiro(3), n_p, n_p * V0)
    st_gk = argmax(st -> st.damage_state.n_flaws, states_gk)
    for (C, broken) in ((1e4 * I3, true), (-1e4 * I3, false))
        st, _, _, σ  = drive(gk, st_gk, t -> C, 1e-6, 50; V0)
        _, _, _, σ_i = drive(intact, initial_material_state(intact), t -> C, 1e-6, 50; V0)
        if broken
            @test st.damage_state.D == 1                        # all flaws active: n_act / n_f = 1
            @test norm(σ) < 1e-6 * norm(σ_i) && tr(σ_i) > 0     # no tension in fragments
        else
            @test st.damage_state.D == 0
            @test σ == σ_i
        end
    end

    # J2 with hypoelasticity (iron, perfect plasticity): pure shear far beyond yield,
    # the stress stays on the yield surface and the plastic work becomes heat
    σ_Y   = 2.5e8
    plast = Iron(Float64; eos = :tillotson, elasticity = :hypoelastic, plasticity = J2Plasticity(σ_y0 = σ_Y))
    elast = Iron(Float64; eos = :tillotson, elasticity = :hypoelastic)
    shear(t) = 1e3 * @SMatrix [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    st_p, _, _, σ_p = drive(plast, initial_material_state(plast), shear, 1e-5, 100)
    st_e, _, _, σ_e = drive(elast, initial_material_state(elast), shear, 1e-5, 100)
    @test von_mises(σ_e) > 10σ_Y                                # loaded far beyond yield ...
    @test von_mises(σ_p) ≈ σ_Y rtol = 1e-10                     # ... but plastic stays on it
    @test st_p.plastic_state.eps_p > 0
    @test st_p.e > 1e5 * abs(st_e.e)                            # dissipation only with plasticity
end

@testset "SolidMaterial: fractured particles" begin
    V0, dx, dt = 1e-6, 1e-3, 1e-8
    dm  = GradyKippDamage{Float64}(E = 53.1e9, m = 8.5, k = 3.0e39)
    mat = Basalt(Float64; damage = dm, viscosity = BulkViscosity())
    nov = Basalt(Float64; damage = dm)
    m   = mat.ρ * V0
    st0 = initial_material_state(mat)
    stf = with_damage(st0, 1.0)
    C   = -1e4 * I3                                           # recompaction

    # F of a fractured particle carries no shear information: F -> J^⅓ I, J ≤ J_max
    Fa = @SMatrix [3.0 0.5 0.0; 0.0 0.5 0.0; 0.1 0.0 1.2]
    @test S.regularize_deformation(mat, stf, Fa) ≈ cbrt(det(Fa)) * I3
    @test S.regularize_deformation(mat, st0, Fa) == Fa        # intact: unchanged
    @test det(S.regularize_deformation(mat, stf, 50.0 * I3)) ≈ dm.J_max

    # artificial viscosity is off in expanded fragments (J > 1), on in compression
    σ(mt, F) = first(material_model(mt, stf, (I3 + C * dt) * F, C, V0, m, dt, dx))
    @test σ(mat, cbrt(2.0) * I3) ≈ σ(nov, cbrt(2.0) * I3)
    @test tr(σ(mat, 0.9 * I3)) < tr(σ(nov, 0.9 * I3))

    # clamped J does no volume work; J > 1 uses the reference sound speed
    _, st1 = material_model(mat, with_energy(stf, 5e7), (I3 + 1e4dt * I3) * cbrt(dm.J_max) * I3, 1e4 * I3, V0, m, dt, dx)
    @test st1.e ≈ 5e7
    _, st2 = material_model(mat, st0, 1.3 * I3, zero(M3), V0, m, 0.0, dx)
    @test st2.c ≈ sqrt(S.reference_soundspeed(mat.eos)^2 + 4 * 2.31e10 / (3 * mat.ρ / 1.3^3))

    # extreme compression in one step stays finite
    Cx = -2e6 * I3
    _, st3 = material_model(mat, st0, I3 + Cx * 1e-7, Cx, V0, m, 1e-7, dx)
    @test isfinite(st3.e) && st3.e > 0
end