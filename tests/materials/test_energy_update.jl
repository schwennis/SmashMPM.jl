@testset "Energy update, fracture regularization" begin
    I3 = one(SMatrix{3,3,Float64,9})
    V0 = 1.0e-6
    dx = 1.0e-3

    # Drives one particle with a prescribed velocity gradient C(t), exactly like the
    # G2P2G kernel does (trial F -> material model -> regularized F is stored).
    # Returns the final state and the external work Σ ½(τ_n + τ_{n+1}) : D dt V0.
    function drive(mat, st, Cfun, t_end, n)
        dt = t_end / n
        m  = mat.ρ * V0
        F  = I3
        σ_old = zero(SMatrix{3,3,Float64,9}); J_old = 1.0
        work = 0.0
        for i in 1:n
            C  = Cfun((i - 0.5) * dt)
            Ft = (I3 + C * dt) * F
            σ, st_new = material_model(mat, st, Ft, C, V0, m, dt, dx)
            F  = SmashMPM.regularize_deformation(mat, st, Ft)
            D  = (C + C') / 2
            work += 0.5 * dt * dot(J_old * σ_old + det(F) * σ, D) * V0
            σ_old, J_old, st = σ, det(F), st_new
        end
        return st, F, work
    end

    @testset "Trapezoidal p: closed compression cycle, 2nd order" begin
        # Tillotson, no strength: compress and expand back to J = 1 (isentropic loop -> e = 0)
        mat = SolidMaterial(eos = Basalt(Float64).eos, elasticity = HypoElasticity{Float64}(0.0),
                            ρ = 2790.0)
        st0 = initial_material_state(mat)
        # J(t) = 1 - 0.1 sin²(π t / t_end): closed loop, hit exactly by det(I + C dt) = J_{i+1} / J_i
        function cycle(n)
            dt, st, J = 1e-5 / n, st0, 1.0
            e_max = 0.0
            for i in 1:n
                J_new = 1 - 0.1 * sin(π * i / n)^2
                C  = ((cbrt(J_new / J) - 1) / dt) * I3
                _, st = material_model(mat, st, cbrt(J_new) * I3, C, V0, mat.ρ * V0, dt, dx)
                J, e_max = J_new, max(e_max, st.e)
            end
            return abs(st.e), e_max
        end
        (e1, e_max), (e2, _) = cycle(100), cycle(200)
        @test e1 < 1e-4 * e_max                         # isentropic loop -> e returns to 0
        @test e1 / e2 > 3.0                             # second order (explicit p_n: ≈ 2)
    end

    @testset "Elastic shear cycle produces (almost) no heat" begin
        for el in (HyperElasticity{Float64}(2.31e10), HypoElasticity{Float64}(2.31e10))
            mat = SolidMaterial(eos = MurnaghanEOS{Float64}(2790.0, 26.7e9, 5.5, 0.9),
                                elasticity = el, ρ = 2790.0)
            st0 = initial_material_state(mat)
            γ̇ = 50.0
            Cf(t) = γ̇ * sin(2π * t / 1e-4) * @SMatrix [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
            st, _, _ = drive(mat, st0, Cf, 1e-4, 400)
            W_max = (2.31e10 * (γ̇ * 1e-4 / π)^2 / 2) / 2790.0   # rough peak elastic energy per mass
            @test abs(st.e) < 1e-3 * W_max
        end
    end

    @testset "Plastic heat: energy balance and positivity" begin
        mat = SolidMaterial(eos = MurnaghanEOS{Float64}(2790.0, 26.7e9, 5.5, 0.9),
                            elasticity = HyperElasticity{Float64}(2.31e10),
                            plasticity = J2Plasticity{Float64}(σ_y0 = 1e8), ρ = 2790.0)
        st0 = initial_material_state(mat)
        Cf(t) = 2e3 * @SMatrix [0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]     # isochoric: no p work
        st, F, work = drive(mat, st0, Cf, 1e-4, 1000)
        W_el = SmashMPM.elastic_energy(mat.elasticity, st.elastic_state, det(F)) * V0
        @test st.e > 0
        # external work = heat + stored elastic energy (up to the p-work of det F drift)
        @test work ≈ mat.ρ * V0 * st.e + W_el rtol = 1e-3
    end

    # Fractured basalt particle
    dm  = GradyKippDamage{Float64}(E = 53.1e9, m = 8.5, k = 3.0e39, n_tot = 1.0e9)
    mat = Basalt(Float64, eos = :tillotson, elasticity = :hypoelastic, damage = dm,
                 viscosity = BulkViscosity())
    st0 = initial_material_state(mat)
    fractured(st) = SmashMPM.SolidMaterialState(c = st.c, e = st.e, eos_state = st.eos_state,
        elastic_state = st.elastic_state, plastic_state = st.plastic_state,
        damage_state = GradyKippState(1.0, 1e-4, 8.5, 1e6))
    stf = fractured(st0)
    m   = mat.ρ * V0
    dt  = 1e-8
    C   = -1e4 * I3                                     # recompaction

    @testset "Isotropic F for fractured particles, J ≤ J_max" begin
        Fa = @SMatrix [3.0 0.5 0.0; 0.0 0.5 0.0; 0.1 0.0 1.2]
        Fr = SmashMPM.regularize_deformation(mat, stf, Fa)
        @test Fr ≈ cbrt(det(Fa)) * I3
        @test SmashMPM.regularize_deformation(mat, st0, Fa) == Fa      # intact: unchanged
        @test det(SmashMPM.regularize_deformation(mat, stf, 50.0 * I3)) ≈ dm.J_max
    end

    @testset "No q in fractured, expanded material" begin
        F = cbrt(2.0) * I3                              # J = 2, compressing again
        σ_f, _ = material_model(mat, stf, (I3 + C * dt) * F, C, V0, m, dt, dx)
        mat_nv = Basalt(Float64, eos = :tillotson, elasticity = :hypoelastic, damage = dm)
        σ_nv, _ = material_model(mat_nv, stf, (I3 + C * dt) * F, C, V0, m, dt, dx)
        @test σ_f ≈ σ_nv                                # q switched off
        σ_c, _ = material_model(mat, stf, (I3 + C * dt) * (0.9 * I3), C, V0, m, dt, dx)
        σ_cn, _ = material_model(mat_nv, stf, (I3 + C * dt) * (0.9 * I3), C, V0, m, dt, dx)
        @test tr(σ_c) < tr(σ_cn)                        # J < 1: q still active
    end

    @testset "Clamped J does no volume work" begin
        F = dm.J_max^(1/3) * I3                         # already at J_max, keeps expanding
        Ce = 1e4 * I3
        st_hot = SmashMPM.SolidMaterialState(c = stf.c, e = 5e7, eos_state = stf.eos_state,
            elastic_state = stf.elastic_state, plastic_state = stf.plastic_state,
            damage_state = stf.damage_state)
        _, st1 = material_model(mat, st_hot, (I3 + Ce * dt) * F, Ce, V0, m, dt, dx)
        @test st1.e ≈ st_hot.e
    end

    @testset "Robust energy update: c_0 for J > 1, safe denominator" begin
        _, st1 = material_model(mat, st0, 1.3 * I3, zero(C), V0, m, 0.0, dx)
        c0 = sqrt(SmashMPM.reference_soundspeed(mat.eos)^2 + 4 * 2.31e10 / (3 * mat.ρ / 1.3^3))
        @test st1.c ≈ c0
        # extreme compression in one step: finite energy
        Cx = -2e6 * I3
        _, st2 = material_model(mat, st0, I3 + Cx * 1e-7, Cx, V0, m, 1e-7, dx)
        @test isfinite(st2.e) && st2.e > 0
        # dt = 0 (export): state unchanged
        _, st3 = material_model(mat, stf, 1.1 * I3, zero(C), V0, m, 0.0, dx)
        @test st3.e == stf.e
    end
end
