
# --- Setup ---
T = Float64
pos = SVector{3, T}(1.0, 2.0, 3.0)
mass = 2.5
vol = 0.5

mat = NeoHookean(E=210e9, ν=0.3, ρ=7850.0)
mat_state = get_initial_material_state(mat)

@testset "Constructors & Type Inference" begin
    # Test particle creation and automatic identity matrix assignment for F
    p = @inferred Particle(pos, mass, vol, mat_state)
    @test p.F == one(SMatrix{3,3,T,9})
    @test p.mass === mass

    # Test material constructor validation
    @test mat.c > 0.0
    @test_throws ErrorException NeoHookean(E=210e9) # Fails because ρ is missing
end

@testset "SoAParticleSet & Type Stability & Memory Layout" begin
    p = Particle(pos, mass, vol, mat_state)
    particles_vec = [p, p, p]
    
    soa_set = SoAParticleSet(particles_vec, mat, KernelAbstractions.CPU())

    @inferred SoAParticleSet(particles_vec, mat, KernelAbstractions.CPU()) # Type stability check for SoAParticleSet constructor
    
    # Verify that the StructArray successfully unwrapped the SVectors (AoS -> SoA)
    # This ensures contiguous memory access for high performance
    @test soa_set.particles.pos isa StructArrays.StructArray
    @test length(soa_set.particles) == 3
    @test length(soa_set.particles_buffer) == 3
end

@testset "Material Model Performance (JET & Type Stability)" begin
    F = one(SMatrix{3,3,T,9})
    C = nothing # Placeholder
    dt = 1e-4

    # Functional correctness & type inference
    @test_nowarn material_model(mat, mat_state, F, C, vol, mass, dt)
    σ, new_state = @inferred material_model(mat, mat_state, F, C, vol, mass, dt)
    @test σ isa SMatrix{3,3,T,9}

    # JET analysis: The material model must NOT contain dynamic dispatches or allocations
    @test_call material_model(mat, mat_state, F, C, vol, mass, dt)
    @test_opt material_model(mat, mat_state, F, C, vol, mass, dt)
end
