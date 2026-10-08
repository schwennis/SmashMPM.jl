@testset "SolidMaterial Pipeline Integration" begin
    # Test am Beispiel Basalt mit Tillotson EOS
    mat = Basalt(Float64, eos=:tillotson, elasticity=:hypoelastic, viscosity=BulkViscosity())
    st0 = initial_material_state(mat)

    V0 = 1.0e-6
    m  = mat.ρ * V0
    dt = 1.0e-7
    dx = 1.0e-3
    I3 = one(SMatrix{3,3,Float64,9})

    @testset "1. Energie-Erhaltung im geschlossenen Zeitschritt" begin
        # Starke Kompression mit Scherung
        C = @SMatrix [-100.0  10.0   0.0;
                       10.0 -100.0   0.0;
                        0.0    0.0 -100.0]
        F = I3 + C * dt

        σ, st1 = material_model(mat, st0, F, C, V0, m, dt, dx)

        # Die interne Energie e muss bei Kompression gestiegen sein
        @test st1.e > st0.e

        # Spannungszerlegung prüfen
        q = SmashMPM.viscous_pressure(mat.viscosity, SmashMPM.Kinematics(F, C, V0, m, dt, dx), st0.c)
        expected_σ = st1.elastic_state.s - (st1.eos_state.p + q) * I3
        @test σ ≈ expected_σ
    end

    @testset "2. Konfigurations-Fabriken (Basalt & Iron)" begin
        @test Basalt(Float64, eos=:murnaghan, elasticity=:hyperelastic) isa SolidMaterial
        @test Iron(Float64, eos=:tillotson, elasticity=:hypoelastic) isa SolidMaterial
    end
end