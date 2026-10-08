@testset "Tillotson EOS" begin
    # Typische Parameter für Eisen
    ρ0  = 7874.0
    A   = 128.0e9
    B   = 105.0e9
    α   = 5.0
    β   = 5.0
    E0  = 9.5e6
    Eiv = 2.4e6
    Ecv = 8.67e6
    a   = 0.5
    b   = 0.15

    eos = TillotsonEOS{Float64}(ρ0, A, B, α, β, E0, Eiv, Ecv, a, b)
    state0 = init_state(eos)

    @testset "1. Referenzzustand" begin
        state, c = update_eos(eos, state0, ρ0, 0.0)
        @test state.p ≈ 0.0 atol=1e-6
        @test c ≈ sqrt(A / ρ0)
    end

    @testset "2. Kompression (ρ >= ρ0)" begin
        ρ_comp = 1.05 * ρ0
        state, c = update_eos(eos, state0, ρ_comp, 0.0)
        
        # Für e=0 muss gelten: p = A*μ + B*μ^2 mit μ = ρ/ρ0 - 1
        μ = 0.05
        expected_p = A * μ + B * μ^2
        @test state.p ≈ expected_p
        @test state.p > 0.0
        @test c > sqrt(A / ρ0)
    end

    @testset "3. Kalt-expandiertes Regime (ρ < ρ0, e < Eiv)" begin
        ρ_exp = 0.95 * ρ0
        e_low = 0.5 * Eiv
        state, c = update_eos(eos, state0, ρ_exp, e_low)
        
        # Muss kondensierte Formulierung verwenden
        p_c, _, _ = SmashMPM._tillotson_condensed(eos, ρ_exp, e_low)
        @test state.p ≈ p_c
        @test c > 0.0
    end

    @testset "4. Heiß-expandiertes / Gas-Regime (ρ < ρ0, e > Ecv)" begin
        ρ_exp = 0.8 * ρ0
        e_high = 2.0 * Ecv
        state, c = update_eos(eos, state0, ρ_exp, e_high)
        
        p_e, _, _ = SmashMPM._tillotson_expanded(eos, ρ_exp, e_high)
        @test state.p ≈ p_e
        @test c > 0.0
    end

    @testset "5. Stetigkeit am Übergangsbereich" begin
        ρ_exp = 0.9 * ρ0
        eps_e = 1e-7

        # Übergang bei Eiv: Links- vs. Rechtsgrenzwert
        s_left_iv,  c_left_iv  = update_eos(eos, state0, ρ_exp, Eiv - eps_e)
        s_right_iv, c_right_iv = update_eos(eos, state0, ρ_exp, Eiv + eps_e)
        @test s_left_iv.p ≈ s_right_iv.p rtol=1e-3

        # Übergang bei Ecv: Links- vs. Rechtsgrenzwert
        s_left_cv,  c_left_cv  = update_eos(eos, state0, ρ_exp, Ecv - eps_e)
        s_right_cv, c_right_cv = update_eos(eos, state0, ρ_exp, Ecv + eps_e)
        @test s_left_cv.p ≈ s_right_cv.p rtol=1e-3
    end

    @testset "6. Numerische Differenzierbarkeit & Konsistenz von c" begin
        # Finite-Differenzen-Check für dp/dρ und dp/de im komprimierten Bereich
        ρ_test = 1.08 * ρ0
        e_test = 1.0e6
        h_ρ = 1e-5 * ρ_test
        h_e = 1e-5 * e_test

        p_base, c_code = update_eos(eos, state0, ρ_test, e_test)
        p_drho, _      = update_eos(eos, state0, ρ_test + h_ρ, e_test)
        p_de, _        = update_eos(eos, state0, ρ_test, e_test + h_e)

        dp_dρ_num = (p_drho.p - p_base.p) / h_ρ
        dp_de_num = (p_de.p - p_base.p) / h_e

        # Definition der Schallgeschwindigkeit: c^2 = dp/dρ + (p/ρ^2) * dp/de
        c_squared_expected = dp_dρ_num + (p_base.p / ρ_test^2) * dp_de_num
        @test c_squared_expected > 0.0
        @test c_code ≈ sqrt(c_squared_expected) rtol=1e-3
    end
end