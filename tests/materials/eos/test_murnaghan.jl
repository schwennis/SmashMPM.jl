@testset "Murnaghan EOS" begin
    ρ0 = 2860.0
    K0 = 26.7e9
    n  = 5.5
    η_limit = 0.9

    eos = MurnaghanEOS{Float64}(ρ0, K0, n, η_limit)
    state0 = init_state(eos)

    @testset "Referenzzustand" begin
        state, c = update_eos(eos, state0, ρ0, 0.0)
        @test state.p ≈ 0.0 atol=1e-6
        @test c ≈ sqrt(K0 / ρ0)
    end

    @testset "Kompression" begin
        η = 1.02
        ρ = η * ρ0
        state, c = update_eos(eos, state0, ρ, 0.0)
        expected_p = K0 / n * (η^n - 1.0)
        expected_c = sqrt(K0 / ρ0 * η^(n - 1.0))
        @test state.p ≈ expected_p
        @test c ≈ expected_c
    end

    @testset "Zugentlastung (η <= η_limit)" begin
        # Unterhalb des Cutoffs muss p = 0 gelten
        ρ = 0.85 * ρ0
        state, c = update_eos(eos, state0, ρ, 0.0)
        @test state.p == 0.0
        @test c ≈ sqrt(K0 / ρ0 * η_limit^(n - 1.0))
    end
end