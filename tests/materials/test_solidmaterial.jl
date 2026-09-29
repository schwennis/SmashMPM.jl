
@testset "SolidMaterial - Typsicherheit und Typstabilität" begin

    # -------------------------------------------------------------------------
    # 1. Strukturelle Typprüfung und Konkretheit
    # -------------------------------------------------------------------------
    @testset "Konkrete Datentypen & Speicherlayout" begin
        # Standardwerte definieren
        density = 2700.0        # kg/m^3 (z. B. Aluminium)
        youngs_modulus = 70.0e9  # Pa
        poisson_ratio = 0.33

        mat64 = SolidMaterial(
            density = density,
            E = youngs_modulus,
            nu = poisson_ratio
        )

        # Prüfen, ob die instanziierte Struktur vollkommen konkret ist (keine abstrakten Typen)
        @test isconcretetype(typeof(mat64))
        
        # Sicherstellen, dass keine Typ-Instabilität durch Typ-Union oder Any in den Feldern vorliegt
        for field_type in fieldtypes(typeof(mat64))
            @test isconcretetype(field_type)
            @test field_type != Any
        end

        # Für Hochleistungs-MPM / GPU-Transfer: Prüfen, ob die Struktur ein Bitstype ist
        @test isbitstype(typeof(mat64))
    end

    # -------------------------------------------------------------------------
    # 2. Präzisionserhalt und Parametrisierung (Float32 / Float64)
    # -------------------------------------------------------------------------
    @testset "Typ-Parametrisierung (Float32 / Float64)" begin
        # Test für Float64
        mat_f64 = SolidMaterial(2500.0, 50.0e9, 0.25)
        @test mat_f64 isa SolidMaterial{Float64}
        @test typeof(mat_f64.density) === Float64
        @test typeof(mat_f64.E) === Float64
        @test typeof(mat_f64.nu) === Float64

        # Test für Float32 (wichtig für GPU-Beschleunigung / Speicherbandbreite)
        mat_f32 = SolidMaterial(2500.0f0, 50.0f9, 0.25f0)
        @test mat_f32 isa SolidMaterial{Float32}
        @test typeof(mat_f32.density) === Float32
        @test typeof(mat_f32.E) === Float32
        @test typeof(mat_f32.nu) === Float32

        # Typenmischung (Mixed-Precision Handling):
        # Ganze Zahlen oder gemischte Floats sollten sauber ohne dynamic dispatch konvertiert werden
        mat_mixed = SolidMaterial(2500, 50.0e9, 0.25)
        @test mat_mixed isa SolidMaterial{Float64}
    end

    # -------------------------------------------------------------------------
    # 3. Typinferenz & Typstabilität der Berechnungsfunktionen (@inferred)
    # -------------------------------------------------------------------------
    @testset "Typinferenz bei Getter- & Materialfunktionen" begin
        mat = SolidMaterial(density = 2700.0, E = 70.0e9, nu = 0.3)

        # Konstruktor-Inferenz testen
        @test_nowarn @inferred SolidMaterial(2700.0, 70.0e9, 0.3)
        @test_nowarn @inferred SolidMaterial(2700.0f0, 70.0f9, 0.3f0)

        # Elastische Kennwerte & Lamé-Parameter (müssen type-stable sein)
        if isdefined(SmashMPM, :lame_lambda)
            @test (@inferred SmashMPM.lame_lambda(mat)) isa Float64
        end
        if isdefined(SmashMPM, :shear_modulus)
            @test (@inferred SmashMPM.shear_modulus(mat)) isa Float64
        end
        if isdefined(SmashMPM, :bulk_modulus)
            @test (@inferred SmashMPM.bulk_modulus(mat)) isa Float64
        end

        # Schallgeschwindigkeiten (Longitudinal- und Transversalwellen)
        if isdefined(SmashMPM, :wave_speed)
            cp = @inferred SmashMPM.wave_speed(mat)
            @test cp isa Float64
            @test cp > 0.0
        end
    end

    # -------------------------------------------------------------------------
    # 4. Invarianten, Gültigkeitsbereiche und Exception-Handling
    # -------------------------------------------------------------------------
    @testset "Validierung unzulässiger Typen & Werte" begin
        # Physikalisch unzulässige Werte müssen gezielt abgewiesen werden
        # Negative Dichte
        @test_throws ArgumentError SolidMaterial(-1000.0, 10.0e9, 0.3)
        
        # Negativer Elastizitätsmodul
        @test_throws ArgumentError SolidMaterial(1000.0, -10.0e9, 0.3)

        # Unzulässige Querkontraktionszahl (Singularität / thermodynamische Stabilität: -1 < nu < 0.5)
        @test_throws ArgumentError SolidMaterial(1000.0, 10.0e9, 0.5)
        @test_throws ArgumentError SolidMaterial(1000.0, 10.0e9, -1.1)

        # Übergabe von Strings oder inkompatiblen Objekten
        @test_throws MethodError SolidMaterial("2700", 70.0e9, 0.3)
    end

    # -------------------------------------------------------------------------
    # 5. Konstitutive Spannungsberechnung / Stress-Update
    # -------------------------------------------------------------------------
    @testset "Typstabilität des Spannungs-Updates" begin
        if isdefined(SmashMPM, :compute_stress)
            using LinearAlgebra
            mat = SolidMaterial(2700.0, 70.0e9, 0.3)
            
            # 3D Deformationsgradient F oder Dehnungstensor epsilon
            strain_tensor = SymmetricTensor{2, 3, Float64}(zeros(3, 3))
            
            # Überprüfen, ob die Funktion ohne dynamische Allokation type-stable ist
            stress = @inferred SmashMPM.compute_stress(mat, strain_tensor)
            @test eltype(stress) === Float64
        end
    end

end