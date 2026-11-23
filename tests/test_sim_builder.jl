

# ------------------------------------------------------------------------ #
# 1. Setup & Konfiguration Tests
# ------------------------------------------------------------------------ #
@testset "SimulationSetup Initialization & Type Stability" begin
    # Test Float64
    setup_f64 = SimulationSetup(dx=Float64(0.1), t_max=Float64(1.0))
    @test setup_f64.dx === 0.1
    @test typeof(setup_f64.cfl_number) == Float64
    
    # Test Float32 (Wichtig für GPU Memory)
    setup_f32 = SimulationSetup(dx=Float32(0.1), t_max=Float32(1.0), cfl_number=Float32(0.4))
    @test setup_f32.dx === 0.1f0
    @test typeof(setup_f32.cfl_number) == Float32

    # Typstabilität des Konstruktors selbst prüfen
    @inferred SimulationSetup(dx=Float64(0.05), t_max=Float64(2.0))
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
    setup = SimulationSetup(dx=0.1, t_max=1.0, ppc_1d=2)

    # -- ACHTUNG: Hier deine echten Material-Konstruktoren nutzen --
    # Wir nehmen an, du hast z.B. ein LinearElastic und ein NeoHookean Material
    mat1 = LinearElastic(ρ=1000.0, E=1e6, ν=0.3)
    mat2 = NeoHookean(ρ=2000.0, E=5e6, ν=0.4)

    sphere1 = Sphere(0.5, SVector(0.0, 0.0, 0.0))
    sphere2 = Sphere(0.3, SVector(2.0, 0.0, 0.0))
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
        
        # Prüfe ob das Grid erfolgreich alloziert wurde (Grid Mass > 0 durch initial_p2g!, jetzt in state_read)
        total_mass = sum(model.grid.state_read.mass)
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

        # Test if map(bodies is unrolled perfectly
        model = @inferred build_mpm_model(bodies_hetero, setup)

        @test length(model.particle_sets) == 2
        
        # Prüfen, ob die Tuples die richtigen Typen behalten haben
        @test model.particle_sets[1].material isa typeof(mat1)
        @test model.particle_sets[2].material isa typeof(mat2)
    end
end


# ------------------------------------------------------------------------ #
# 4. Initial half kick (KDK start)
# ------------------------------------------------------------------------ #
# build_mpm_model kicks the initial grid momentum by dt0/2 with the initial
# forces, so state_read holds m * v^{1/2}, and stores dt0 in dt_previous.
@testset "build_mpm_model: initial half kick" begin
    T   = Float64
    g   = SVector{3, T}(0.0, -9.81, 0.0)
    v0  = SVector{3, T}(1.0, 0.0, 0.0)
    mat = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
    body = Body(Sphere(0.3, SVector(0.0, 0.0, 0.0)), v0, zero(SVector{3, T}), mat)

    read_state(grid::DenseGrid)  = grid.state_read
    read_state(grid::SparseGrid) = to_dense(grid).state_read

    for grid_type in (DenseGrid, SparseGrid)
        @testset "$grid_type" begin
            setup = SimulationSetup(dx=0.1, t_max=1.0, grid_type=grid_type,
                                    external_force=ConstantGravity(g))
            model = build_mpm_model((body,), setup)

            @test model isa MPMModel
            @test model.t == 0.0
            @test model.cfl_number == setup.cfl_number      # keyword constructor, no field shift

            # dt_previous is the FULL first time step (not dt0/2)
            dt0 = model.dt_previous
            @test dt0 > 0
            @test dt0 ≈ SmashMPM.courant_timestep(model)    # wave speeds unchanged since build

            st = read_state(model.grid)
            M  = sum(st.mass)
            P  = SVector(sum(st.momentum.x), sum(st.momentum.y), sum(st.momentum.z))

            # P = M v0 + dt0/2 * M g  (internal forces are zero in the unstressed initial state)
            @test P[1] ≈ M * v0[1]               rtol = 1e-10
            @test P[2] ≈ M * g[2] * dt0 / 2      rtol = 1e-10
            @test abs(P[3]) < 1e-10 * M * norm(v0)

            # the force of the initial state is kept in the read state (f = m g)
            @test sum(st.force.y) ≈ M * g[2]     rtol = 1e-10
        end
    end

    # without external forces the initial momentum is not changed by the kick
    model = build_mpm_model((body,), SimulationSetup(dx=0.1, t_max=1.0))
    st = model.grid.state_read
    @test sum(st.momentum.y) ≈ 0.0 atol = 1e-12
    @test sum(st.momentum.x) ≈ sum(st.mass) * v0[1] rtol = 1e-10

    # t_max shorter than the CFL step: the first step is clipped to t_max
    model = build_mpm_model((body,), SimulationSetup(dx=0.1, t_max=1e-6))
    @test model.dt_previous ≈ 1e-6
end
