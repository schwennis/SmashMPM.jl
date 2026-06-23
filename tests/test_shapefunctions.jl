@testset "Grid Position" begin
    # At [0,0,0] with 1/dx = 1, origin at [0,0,0], and padding of 0, we should get [1,1,1] (one-based indexing)
    pos_p = @SVector [0.0, 0.0, 0.0]
    inv_dx = 1.0
    origin = @SVector [0.0, 0.0, 0.0]
    padding = 0

    @test SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding) ≈ @SVector [1.0, 1.0, 1.0]
    @test_opt SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding)
    @test_call SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding)


    # At [0.5,0.5,0.5] with 1/dx = 1, origin at [0,0,0], and padding of 0, we should get [1.5,1.5,1.5]
    pos_p = @SVector [0.5, 0.5, 0.5]
    @test SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding) ≈ @SVector [1.5, 1.5, 1.5]
    @test_opt SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding)
    @test_call SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding)

    # With padding of 1 and origin at [-1,-1,-1], the same position should yield [3.5,3.5,3.5]
    padding = 1
    origin = @SVector [-1.0, -1.0, -1.0]
    @test SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding) ≈ @SVector [3.5, 3.5, 3.5]
    @test_opt SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding)
    @test_call SmashMPM.get_grid_position(pos_p, inv_dx, origin, padding)
end

@testset "Quadratic Shape Functions" begin
    spline = QuadraticSpline()

    @testset "1D Spline Evaluation (quadspline_1d)" begin
        # 1. Exakte Werte an den Knotenpunkten
        @test SmashMPM.quadspline_1d(0.0) == 0.75
        @test SmashMPM.quadspline_1d(0.5) == 0.5
        @test SmashMPM.quadspline_1d(-0.5) == 0.5
        @test SmashMPM.quadspline_1d(1.5) == 0.0
        @test SmashMPM.quadspline_1d(-1.5) == 0.0
        
        # 2. Werte außerhalb des Supports (Kompakter Träger)
        @test SmashMPM.quadspline_1d(1.6) == 0.0
        @test SmashMPM.quadspline_1d(10.0) == 0.0

        # 3. Symmetrie
        @test SmashMPM.quadspline_1d(0.2) == SmashMPM.quadspline_1d(-0.2)
        @test SmashMPM.quadspline_1d(1.1) == SmashMPM.quadspline_1d(-1.1)
    end

    @testset "3D Spline Evaluation (shapefunction)" begin
        # Im Ursprung (0,0,0) muss N = 0.75^3 = 0.421875 sein
        pos_origin = SVector{3, Float64}(0.0, 0.0, 0.0)
        @test shapefunction(spline, pos_origin) == 0.75^3

        # Außerhalb des Supports in einer beliebigen Achse
        pos_out = SVector{3, Float64}(0.0, 1.6, 0.0)
        @test shapefunction(spline, pos_out) == 0.0
    end


    @testset "JET & Type Stability Checks" begin
        pos = SVector{3, Float64}(0.2, -0.1, 0.4)
        
        # 1. Inferierbarkeit prüfen (Kein Any-Typ)
        @inferred shapefunction(spline, pos)
        @inferred SmashMPM.get_support_base(spline, pos)
        
        # 2. Strenge Performance-Analyse mit JET
        @test_opt shapefunction(spline, pos)
        @test_opt SmashMPM.get_support_base(spline, pos)
        
        # APIC JET Checks
        N_val = 0.5
        r_rel = SVector{3, Float64}(1.0, 2.0, 3.0)
        v_I   = SVector{3, Float64}(0.1, 0.2, 0.3)
        
        @test_opt SmashMPM.B_update(spline, N_val, r_rel, v_I)
    end



end