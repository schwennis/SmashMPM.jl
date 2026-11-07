using Random

@testset "Grady-Kipp Damage" begin
    # Basalt, Table 1 in Yan et al. (2026)
    dm = GradyKippDamage{Float64}(E=53.1e9, m=8.5, k=3.0e39, n_tot=1.0e9)
    V   = 4 / 3 * π * 0.03^3
    n_p = 1000

    I3 = one(SMatrix{3,3,Float64,9})
    s0 = zero(SMatrix{3,3,Float64,9})
    kin(dt) = SmashMPM.Kinematics(I3, s0, V / n_p, 2700.0 * V / n_p, dt, 1e-3)
    c  = 5000.0

    @testset "Flaw initialisierung" begin
        rng = Random.Xoshiro(1)
        states = [SmashMPM.init_state(dm, rng, n_p, V) for _ in 1:n_p]
        ε_V = (dm.k * V)^(-1 / dm.m)

        @test all(st.D == 0.0 for st in states)
        @test all(ε_V <= st.ε_min <= 10ε_V for st in states)
        @test sum(st.ε_min for st in states) / n_p ≈ 10dm.m / (dm.m + 1) * ε_V rtol=3e-2
        @test sum(st.n_f for st in states) / n_p ≈ dm.n_tot / n_p rtol=1e-2
        @test all(st.m_p > 0 for st in states)

        @test SmashMPM.init_state(dm).ε_min == Inf
    end

    @testset "Hauptspannung" begin
        σ = @SMatrix [3.0 1.0 0.0; 1.0 3.0 0.0; 0.0 0.0 -2.0]
        @test SmashMPM.max_principal_stress(σ) ≈ 4.0
        @test SmashMPM.max_principal_stress(5.0 * I3) ≈ 5.0
    end

    st = SmashMPM.GradyKippState(0.0, 1e-4, 8.5, 1.0e6)

    @testset "Kein Schaden unter Druck" begin
        th = (p = 1e9, T = 0.0)
        @test SmashMPM.update_damage(dm, st, s0, SmashMPM.NullState(), th, kin(1e-7), c) == st
    end

    @testset "Schadenswachstum unter Zug" begin
        th = (p = -2.0 * dm.E * st.ε_min, T = 0.0)
        st1 = SmashMPM.update_damage(dm, st, s0, SmashMPM.NullState(), th, kin(1e-11), c)
        @test 0.0 < st1.D < 1.0

        n_act = 2.0^st.m_p
        R_s   = cbrt(3 * V / n_p / (4π))
        @test cbrt(st1.D) ≈ 1e-11 * n_act * 0.4c / R_s

        # Obergrenze D_max = (n_act / n_f)^(1/3)
        st2 = SmashMPM.update_damage(dm, st, s0, SmashMPM.NullState(), th, kin(1.0), c)
        @test st2.D ≈ n_act / st.n_f
        @test SmashMPM.update_damage(dm, st2, s0, SmashMPM.NullState(), th, kin(1.0), c).D >= st2.D
    end

    @testset "Degradierung" begin
        st_d = SmashMPM.GradyKippState(0.25, 1e-4, 8.5, 1.0e6)
        @test SmashMPM.degradation(dm, st_d) == 0.75
        @test SmashMPM.degrade_pressure(dm, st_d, -4.0) == -3.0
        @test SmashMPM.degrade_pressure(dm, st_d, 4.0) == 4.0
        @test SmashMPM.damage_variable(st_d) == 0.25
    end

    @testset "SolidMaterial Integration" begin
        mat = Basalt(Float64, eos=:tillotson, elasticity=:hypoelastic, damage=dm)
        st0 = initial_material_state(mat, Random.Xoshiro(2), n_p, V)
        @test st0 isa typeof(initial_material_state(mat))
        @test isfinite(st0.damage_state.ε_min)

        C = 1e4 * I3
        F = I3 + C * 1e-7
        σ, st1 = @inferred material_model(mat, st0, F, C, V / n_p, mat.ρ * V / n_p, 1e-7, 1e-3)
        @test st1.damage_state.D >= st0.damage_state.D
        @test all(isfinite, σ)
    end
end