@testset "Bulk Viscosity" begin
    visc = BulkViscosity(c_quad=1.5, c_lin=0.06)
    c = 3000.0
    dx = 0.01

    I3 = one(SMatrix{3,3,Float64,9})

    @testset "Inaktiv bei Expansion (div(v) >= 0)" begin
        C_exp = 0.1 * I3 # tr(C) = 0.3 > 0
        kin = SmashMPM.Kinematics(I3, C_exp, 1.0, 1000.0, 1e-4, dx)
        q = SmashMPM.viscous_pressure(visc, kin, c)
        @test q == 0.0
    end

    @testset "Aktiv bei Kompression (div(v) < 0)" begin
        divv = -10.0
        C_comp = (divv / 3.0) * I3
        kin = SmashMPM.Kinematics(I3, C_comp, 1.0, 1000.0, 1e-4, dx)
        q = SmashMPM.viscous_pressure(visc, kin, c)

        ρ = 1000.0
        expected_q = ρ * dx * (1.5 * dx * divv^2 - 0.06 * c * divv)
        @test q ≈ expected_q
        @test q > 0.0
    end
end