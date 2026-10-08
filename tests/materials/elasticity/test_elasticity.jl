@testset "Elasticity Models" begin
    μ = 30.0e9
    dt = 1e-4
    dx = 1e-2

    # Dummy Kinematics Konstruktor
    function make_kin(F, C)
        SmashMPM.Kinematics(F, C, 1.0, 1.0, dt, dx)
    end

    I3 = one(SMatrix{3,3,Float64,9})
    zero3 = zeros(SMatrix{3,3,Float64,9})

    @testset "HypoElasticity (Jaumann-Rate)" begin
        el = SmashMPM.HypoElasticity(μ)
        st = SmashMPM.init_state(el)

        # 1. Reine Scherung: C_12 = C_21 = γ_dot
        γ_dot = 10.0
        C = @SMatrix [0.0 γ_dot 0.0;
                      γ_dot 0.0 0.0;
                      0.0   0.0 0.0]
        kin = make_kin(I3, C)
        s_tr, st_tr, G = SmashMPM.elastic_trial(el, st, kin)

        @test G == μ
        @test s_tr[1,2] ≈ 2.0 * μ * dt * γ_dot   # 2μ · dt · D_dev,12
        @test tr(s_tr) ≈ 0.0 atol=1e-12

        # 2. Starrkörperrotation: W != 0, D = 0 -> keine Spannungsänderung bei s = 0
        ω = 5.0
        C_rot = @SMatrix [0.0 -ω 0.0;
                          ω    0.0 0.0;
                          0.0  0.0 0.0]
        kin_rot = make_kin(I3, C_rot)
        s_tr_rot, _, _ = SmashMPM.elastic_trial(el, st, kin_rot)
        @test all(s_tr_rot .≈ 0.0)
    end

    @testset "HyperElasticity (Neo-Hookean Isochoric)" begin
        el = SmashMPM.HyperElasticity(μ)
        st = SmashMPM.init_state(el)

        # Im undeformierten Zustand muss s = 0 sein
        kin_id = make_kin(I3, zero3)
        s_tr, st_new, G = SmashMPM.elastic_trial(el, st, kin_id)
        @test all(s_tr .≈ 0.0)
        @test G ≈ μ

        # Isotropische Streckung det(F) != 1: Abweichung des bbar_e muss spurfrei bleiben
        F_iso = 1.05 * I3
        kin_iso = make_kin(F_iso, zero3)
        s_tr_iso, _, _ = SmashMPM.elastic_trial(el, st, kin_iso)
        @test tr(s_tr_iso) ≈ 0.0 atol=1e-10
    end
end