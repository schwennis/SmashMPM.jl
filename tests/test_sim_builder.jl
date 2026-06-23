

# ------------------------------------------------------------------------ #
# 1. Setup & Konfiguration Tests
# ------------------------------------------------------------------------ #
@testset "SimulationSetup Initialization & Type Stability" begin
    # Test Float64
    setup_f64 = SimulationSetup(dx=0.1, t_end=1.0)
    @test setup_f64.dx === 0.1
    @test typeof(setup_f64.CFL_number) == Float64
    
    # Test Float32 (Wichtig für GPU Memory)
    setup_f32 = SimulationSetup(dx=0.1f0, t_end=1.0f0, CFL_number=0.4f0)
    @test setup_f32.dx === 0.1f0
    @test typeof(setup_f32.CFL_number) == Float32

    # Typstabilität des Konstruktors selbst prüfen
    @inferred SimulationSetup(dx=0.05, t_end=2.0)
end

# ------------------------------------------------------------------------ #
# 2. Bounding Box Tests
# ------------------------------------------------------------------------ #
@testset "Bounding Box Utility" begin
    positions = [
        SVector(0.0, 0.0, 0.0), 
        SVector(1.5, -2.0, 3.0), 
        SVector(-1.0, 5.0, 2.0)
    ]
    
    # 1. Typstabilität prüfen
    min_c, max_c = @inferred SmashMPM.bounding_box(positions)
    
    # 2. Korrektheit prüfen
    @test min_c == SVector(-1.0, -2.0, 0.0)
    @test max_c == SVector(1.5, 5.0, 3.0)
end

# ------------------------------------------------------------------------ #
# 3. Model Build Tests (Der ultimative G2P2G Test)
# ------------------------------------------------------------------------ #
@testset "build_mpm_model Typstabilität" begin
    
    # Erstelle Dummy-Setups
    T = Float64
    setup = SimulationSetup(dx=0.1, t_end=1.0, ppc_1d=2)

    # -- ACHTUNG: Hier deine echten Material-Konstruktoren nutzen --
    # Wir nehmen an, du hast z.B. ein LinearElastic und ein NeoHookean Material
    mat1 = LinearElastic(ρ=1000.0, E=1e6, ν=0.3)
    mat2 = NeoHookean(ρ=2000.0, E=5e6, ν=0.4)

    sphere1 = Sphere(SVector(0.0, 0.0, 0.0), 0.5)
    sphere2 = Sphere(SVector(2.0, 0.0, 0.0), 0.3)
    # Wenn du Zylinder schon drin hast, teste auch verschiedene Shapes!
    # cylinder = Cylinder(SVector(2.0,0.0,0.0), 0.5, 1.0, SVector(0.0,0.0,0.0))

    # -------------------------------------------------------------------- #
    # Test A: Homogenes Tuple (Gleiche Shapes, Gleiches Material)
    # -------------------------------------------------------------------- #
    @testset "Homogeneous Bodies" begin
        body1 = Body(sphere1, SVector(1.0, 0.0, 0.0), SVector(0.0,0.0,0.0), mat1)
        body2 = Body(sphere2, SVector(-1.0, 0.0, 0.0), SVector(0.0,0.0,0.0), mat1)
        
        bodies_homo = (body1, body2)

        # DIE MAGISCHE ZEILE: @inferred prüft, ob die Funktion ohne "Any" auskommt
        model = @inferred build_mpm_model(bodies_homo, setup)

        @test model isa MPMModel
        @test length(model.particle_sets) == 2
        
        # Prüfe ob das Grid erfolgreich alloziert wurde (Grid Mass > 0 durch initial_p2g!)
        total_mass = sum(model.grid.state_old.mass)
        @test total_mass > 0.0
    end

    # -------------------------------------------------------------------- #
    # Test B: Heterogenes Tuple (Verschiedene Materialien / Shapes)
    # -------------------------------------------------------------------- #
    @testset "Heterogeneous Bodies (Multiple Dispatch Unrolling)" begin
        body_elastic    = Body(sphere1, SVector(1.0, 0.0, 0.0), SVector(0.0,0.0,0.0), mat1)
        body_neohookean = Body(sphere2, SVector(0.0, 1.0, 0.0), SVector(0.0,0.0,0.0), mat2)
        
        # Tuple mit ZWEI VERSCHIEDENEN Datentypen
        bodies_hetero = (body_elastic, body_neohookean)

        # Wenn dieser @inferred Test durchläuft, hast du bewiesen, dass dein 
        # `map(bodies)` Konstrukt perfekt entrollt (Loop Unrolling) wird!
        model = @inferred build_mpm_model(bodies_hetero, setup)

        @test length(model.particle_sets) == 2
        
        # Prüfen, ob die Tuples die richtigen Typen behalten haben
        @test model.particle_sets[1].material isa typeof(mat1)
        @test model.particle_sets[2].material isa typeof(mat2)
    end
end
