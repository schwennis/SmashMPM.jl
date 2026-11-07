@testset "J2 Plasticity" begin
    μ   = 105.0e9
    σ_Y = 2.5e8
    dt  = 1e-7
    dx  = 1e-3
    sqrt_2_3 = sqrt(2.0 / 3.0)
    Y0       = sqrt_2_3 * σ_Y           # radius of the initial yield surface in stress space

    I3  = one(SMatrix{3,3,Float64,9})
    kin = SmashMPM.Kinematics(I3, zeros(SMatrix{3,3,Float64,9}), 1.0, 1.0, dt, dx)

    # Trace-free trial stresses of a given Frobenius norm
    shear_dir = (@SMatrix [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]) / sqrt(2.0)
    gen_dir   = (@SMatrix [3.0 1.0 0.5; 1.0 -1.0 2.0; 0.5 2.0 -2.0])
    gen_dir   = gen_dir / norm(gen_dir)

    return_map(pl, st, s_tr) = SmashMPM.return_map(pl, st, s_tr, μ, nothing, kin)

    @testset "State and defaults" begin
        pl = J2Plasticity(σ_y0=σ_Y)
        st = SmashMPM.init_state(pl)

        @test pl.H == 0.0
        @test st isa SmashMPM.J2PlasticState{Float64}
        @test SmashMPM.equivalent_plastic_strain(st) == 0.0
        @test !SmashMPM.damage_variable(st)
    end

    @testset "Elastic step leaves stress and state untouched" begin
        pl = J2Plasticity(σ_y0=σ_Y, H=1.0e9)
        st = SmashMPM.init_state(pl)

        for scale in (0.0, 0.5, 0.999)
            s_tr = scale * Y0 * gen_dir
            s, st_new = return_map(pl, st, s_tr)
            @test s === s_tr
            @test st_new === st
        end

        # Just above the yield surface the step is plastic
        s, st_new = return_map(pl, st, 1.001 * Y0 * gen_dir)
        @test st_new.eps_p > 0
        @test norm(s) < 1.001 * Y0
    end

    @testset "Perfect plasticity (H = 0)" begin
        pl   = J2Plasticity(σ_y0=σ_Y)
        st   = SmashMPM.init_state(pl)
        s_tr = 2Y0 * gen_dir
        s, st_new = return_map(pl, st, s_tr)

        # Radial return: back onto the yield surface along the trial direction
        @test norm(s) ≈ Y0 rtol = 1e-12
        @test s ≈ s_tr * (norm(s) / norm(s_tr)) rtol = 1e-12
        @test tr(s) ≈ 0.0 atol = 1e-6

        # α = √(2/3) Δγ with Δγ = f_tr / 2μ
        @test st_new.eps_p ≈ sqrt_2_3 * (norm(s_tr) - Y0) / 2μ rtol = 1e-12
    end

    @testset "Linear isotropic hardening" begin
        alphas = Float64[]
        for H in (1.0e9, 5.0e9, 5.0e10)
            pl   = J2Plasticity(σ_y0=σ_Y, H=H)
            st   = SmashMPM.init_state(pl)
            s_tr = 3Y0 * shear_dir
            s, st_new = return_map(pl, st, s_tr)

            # Consistency condition f(s_{n+1}, α_{n+1}) = 0
            @test norm(s) ≈ sqrt_2_3 * (σ_Y + H * st_new.eps_p) rtol = 1e-12
            # Closed form: α = √(2/3) f_tr / (2μ + 2H/3)
            @test st_new.eps_p ≈ sqrt_2_3 * (norm(s_tr) - Y0) / (2μ + 2H / 3) rtol = 1e-12
            push!(alphas, st_new.eps_p)
        end
        # Stiffer hardening leaves less plastic strain
        @test issorted(alphas, rev=true)
    end

    @testset "Hardening is remembered by the state" begin
        H    = 1.0e10
        pl   = J2Plasticity(σ_y0=σ_Y, H=H)
        s_tr = 1.5Y0 * gen_dir

        # Plastic for a virgin material ...
        _, st_virgin = return_map(pl, SmashMPM.init_state(pl), s_tr)
        @test st_virgin.eps_p > 0

        # ... but elastic once α_n has doubled the yield stress
        st_hard = SmashMPM.J2PlasticState(σ_Y / H)
        s, st_new = return_map(pl, st_hard, s_tr)
        @test s === s_tr
        @test st_new === st_hard
    end

    @testset "Proportional loading is path independent" begin
        pl = J2Plasticity(σ_y0=σ_Y, H=2.0e10)
        st = SmashMPM.init_state(pl)
        N1, ΔN = 2Y0, 1.5Y0

        # One step to N1 + ΔN ...
        s_direct, st_direct = return_map(pl, st, (N1 + ΔN) * gen_dir)

        # ... equals two steps, the second one starting from the plastic state
        s1, st1 = return_map(pl, st, N1 * gen_dir)
        s2, st2 = return_map(pl, st1, s1 + ΔN * gen_dir)

        @test s2 ≈ s_direct rtol = 1e-12
        @test st2.eps_p ≈ st_direct.eps_p rtol = 1e-12
    end

    @testset "Returned state is consistent (no further flow)" begin
        for H in (0.0, 3.0e9)
            pl = J2Plasticity(σ_y0=σ_Y, H=H)
            s1, st1 = return_map(pl, SmashMPM.init_state(pl), 2.5Y0 * gen_dir)
            s2, st2 = return_map(pl, st1, s1)

            @test s2 ≈ s1 rtol = 1e-10
            @test st2.eps_p ≈ st1.eps_p rtol = 1e-10
        end
    end

    @testset "Precision and type stability" begin
        I3_32 = one(SMatrix{3,3,Float32,9})
        kin32 = SmashMPM.Kinematics(I3_32, zeros(SMatrix{3,3,Float32,9}), 1.0f0, 1.0f0, 1.0f-7, 1.0f-3)
        s_tr  = Float32(2Y0) * SMatrix{3,3,Float32,9}(gen_dir)
        μ32   = Float32(μ)

        pl32 = J2Plasticity(σ_y0=Float32(σ_Y), H=1.0f9)
        st32 = SmashMPM.init_state(pl32)
        s, st_new = @inferred SmashMPM.return_map(pl32, st32, s_tr, μ32, nothing, kin32)
        @test eltype(s) == Float32
        @test st_new isa SmashMPM.J2PlasticState{Float32}
        @test st_new.eps_p > 0

        # Float64 parameters with Float32 kinematics: computed in Float32, state type is kept
        pl64 = J2Plasticity(σ_y0=σ_Y, H=1.0e9)
        st64 = SmashMPM.init_state(pl64)
        s, st_new = @inferred SmashMPM.return_map(pl64, st64, s_tr, μ32, nothing, kin32)
        @test eltype(s) == Float32
        @test st_new isa SmashMPM.J2PlasticState{Float64}
    end

    @testset "Integration in SolidMaterial (pure shear)" begin
        mat_pl = Iron(Float64; eos=:tillotson, elasticity=:hypoelastic, plasticity=J2Plasticity(σ_y0=σ_Y))
        mat_el = Iron(Float64; eos=:tillotson, elasticity=:hypoelastic)

        V0 = 1.0e-6
        m  = mat_pl.ρ * V0
        γ̇  = 1.0e3                      # shear rate D_12 = D_21
        C  = @SMatrix [0.0 γ̇ 0.0; γ̇ 0.0 0.0; 0.0 0.0 0.0]
        n_steps = 40

        st_pl = initial_material_state(mat_pl)
        st_el = initial_material_state(mat_el)
        for _ in 1:n_steps
            _, st_pl = material_model(mat_pl, st_pl, I3, C, V0, m, dt, dx)
            _, st_el = material_model(mat_el, st_el, I3, C, V0, m, dt, dx)
        end

        # Without plasticity the stress grows far beyond the yield surface ...
        @test norm(st_el.elastic_state.s) > 2Y0
        @test SmashMPM.equivalent_plastic_strain(st_el.plastic_state) == 0

        # ... with plasticity it saturates exactly on it
        s = st_pl.elastic_state.s
        @test norm(s) ≈ Y0 rtol = 1e-10
        @test tr(s) ≈ 0.0 atol = 1e-6

        # Proportional loading: total strain = elastic part + plastic part, α = √(2/3) ‖ε_p‖
        strain_norm = sqrt(2.0) * γ̇ * dt * n_steps
        @test st_pl.plastic_state.eps_p ≈ sqrt_2_3 * (strain_norm - norm(s) / 2μ) rtol = 1e-8
    end
end