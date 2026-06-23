@testset "Constructor & Parameters" begin
    # 1. Test for missing density (ρ)
    @test_throws ErrorException NeoHookean(E=1.0, ν=0.3)
    
    # 2. Test for incomplete material parameters (neither E/ν nor λ/μ)
    @test_throws ErrorException NeoHookean(ρ=1000.0)
    
    # 3. Direct initialization with Lamé parameters (λ, μ)
    # --- JET Checks ---
    @test_opt NeoHookean(λ=100.0, μ=50.0, ρ=2.0)
    @test_call NeoHookean(λ=100.0, μ=50.0, ρ=2.0)
    # ------------------
    mat_lame = NeoHookean(λ=100.0, μ=50.0, ρ=2.0)
    @test mat_lame.λ == 100.0
    @test mat_lame.μ == 50.0
    @test mat_lame.ρ == 2.0
    @test mat_lame.c == sqrt((100.0 + 2 * 50.0) / 2.0) # sqrt(200/2) = 10.0

    # 4. Initialization via Young's modulus (E) and Poisson's ratio (ν)
    E_val = 210e9  # e.g., steel in Pascals
    ν_val = 0.3
    ρ_val = 7800.0
    
    # --- JET Checks ---
    @test_opt NeoHookean(E=E_val, ν=ν_val, ρ=ρ_val)
    @test_call NeoHookean(E=E_val, ν=ν_val, ρ=ρ_val)
    # ------------------
    mat_eng = NeoHookean(E=E_val, ν=ν_val, ρ=ρ_val)
    
    # Manually calculate reference values
    expected_λ = E_val * ν_val / ((1 + ν_val) * (1 - 2 * ν_val))
    expected_μ = E_val / (2 * (1 + ν_val))
    expected_c = sqrt((expected_λ + 2 * expected_μ) / ρ_val)

    # Always use ≈ instead of == for floating-point operations
    @test mat_eng.λ ≈ expected_λ
    @test mat_eng.μ ≈ expected_μ
    @test mat_eng.c ≈ expected_c
end

@testset "State & Soundspeed" begin
    mat = NeoHookean(λ=100.0, μ=50.0, ρ=2.0)
    
    # --- JET Checks ---
    @test_opt get_initial_material_state(mat)
    @test_call get_initial_material_state(mat)
    # ------------------
    state = get_initial_material_state(mat)
    
    # Check if the correct state type is returned
    @test state isa NoMaterialState
    
    # --- JET Checks ---
    @test_opt get_soundspeed(mat, state)
    @test_call get_soundspeed(mat, state)
    # ------------------
    # Check if the sound speed is read correctly from the struct
    @test get_soundspeed(mat, state) == mat.c
end

@testset "Material Model (Stress)" begin
    mat = NeoHookean(λ=10.0, μ=5.0, ρ=1.0)
    state = get_initial_material_state(mat)
    
    # Dummy values for parameters currently ignored by your NeoHookean model
    C_dummy = nothing
    V0_dummy = 1.0
    m_dummy = 1.0
    dt_dummy = 0.1

    @testset "Identity Deformation" begin
        # F = Identity matrix (No deformation)
        F_id = @SMatrix [1.0 0.0 0.0; 
                0.0 1.0 0.0; 
                0.0 0.0 1.0]
        
        # --- JET Checks ---
        @test_opt material_model(mat, state, F_id, C_dummy, V0_dummy, m_dummy, dt_dummy)
        @test_call material_model(mat, state, F_id, C_dummy, V0_dummy, m_dummy, dt_dummy)
        # ------------------
        
        σ_id, new_state = material_model(mat, state, F_id, C_dummy, V0_dummy, m_dummy, dt_dummy)
        
        # For J=1 (log(J)=0) and b=I, the Cauchy stress σ must be exactly 0
        @test all(σ_id .≈ 0.0)
        @test new_state isa NoMaterialState
    end

    @testset "Simple Stretch (Uniaxial)" begin
        # Deformation only in x-direction (stretch by factor of 2)
        F_stretch = @SMatrix [2.0 0.0 0.0; 
                     0.0 1.0 0.0; 
                     0.0 0.0 1.0]
                        
        # --- JET Checks ---
        @test_opt material_model(mat, state, F_stretch, C_dummy, V0_dummy, m_dummy, dt_dummy)
        @test_call material_model(mat, state, F_stretch, C_dummy, V0_dummy, m_dummy, dt_dummy)
        # ------------------
        
        σ_stretch, _ = material_model(mat, state, F_stretch, C_dummy, V0_dummy, m_dummy, dt_dummy)
        
        # --- Manual precalculation ---
        # b = F * F' = diag([4.0, 1.0, 1.0])
        # J = det(F) = 2.0
        # log(J) = log(2.0)
        # σ_xx = (μ * (b_xx - 1) + λ * log(J)) / J 
        #      = (5.0 * 3.0 + 10.0 * log(2.0)) / 2.0
        # σ_yy = σ_zz = (μ * (1 - 1) + λ * log(J)) / J
        #      = (10.0 * log(2.0)) / 2.0
        
        expected_σ_xx = (15.0 + 10.0 * log(2.0)) / 2.0
        expected_σ_yy = (10.0 * log(2.0)) / 2.0
        expected_σ_zz = expected_σ_yy
        
        # Check diagonal
        @test σ_stretch[1, 1] ≈ expected_σ_xx
        @test σ_stretch[2, 2] ≈ expected_σ_yy
        @test σ_stretch[3, 3] ≈ expected_σ_zz
        
        # Shear stresses (off-diagonals) must be 0
        @test σ_stretch[1, 2] == 0.0
        @test σ_stretch[1, 3] == 0.0
        @test σ_stretch[2, 3] == 0.0
    end
end