@testset "Constructor & Initial State" begin
	eos = MurnaghanEOS(ρ=1.0, K0=100.0, n=2.0, η_limit=0.9)
	elasticity = SmashMPM.HypoElasticity(μ=30.0)
	material = SmashMPM.SolidMaterial(eos, elasticity, SmashMPM.NoPlasticity(), SmashMPM.NoDamage(), SmashMPM.NoViscosity(), 1.0)

	@test material.eos === eos
	@test material.elasticity === elasticity
	@test material.ρ == 1.0

	state = get_initial_material_state(material)

	@test state isa SolidMaterialState{Float64}
	@test state.eos_state isa MurnaghanState{Float64}
	@test state.elastic_state isa SmashMPM.HypoElasticState{Float64}
	@test state.eos_state.p == 0.0
	@test state.elastic_state.s == zeros(SMatrix{3, 3, Float64, 9})
	@test get_soundspeed(material, state) ≈ sqrt(100.0 + 4.0 * 30.0 / 3.0)
end

@testset "Material Model" begin
	eos = MurnaghanEOS(ρ=1.0, K0=100.0, n=2.0, η_limit=0.9)
	elasticity = SmashMPM.HypoElasticity(μ=30.0)
	material = SmashMPM.SolidMaterial(eos, elasticity, SmashMPM.NoPlasticity(), SmashMPM.NoDamage(), SmashMPM.NoViscosity(), 1.0)
	state = get_initial_material_state(material)
	identity = one(SMatrix{3, 3, Float64, 9})
	zero_gradient = zeros(SMatrix{3, 3, Float64, 9})

	@testset "Reference Configuration" begin
		stress, new_state = material_model(material, state, identity, zero_gradient, 1.0, 1.0, 0.01, 0.01)

		@test stress == zeros(SMatrix{3, 3, Float64, 9})
		@test new_state.eos_state.p == 0.0
		@test new_state.elastic_state.s == zeros(SMatrix{3, 3, Float64, 9})
		@test get_soundspeed(material, new_state) ≈ sqrt(100.0 + 4.0 * 30.0 / 3.0)
	end

	@testset "Hydrostatic Compression" begin
		deformation = @SMatrix [0.5 0.0 0.0;
								0.0 1.0 0.0;
								0.0 0.0 1.0]
		stress, new_state = material_model(material, state, deformation, zero_gradient, 1.0, 1.0, 0.01, 0.01)

		expected_pressure = 100.0 / 2.0 * (2.0^2 - 1.0)
		expected_stress = -expected_pressure * identity
		expected_soundspeed = sqrt(4.0 * 30.0 / (3.0 * 2.0) + 100.0 / 1.0 * 2.0^2 / 2.0)

		@test stress ≈ expected_stress
		@test new_state.eos_state.p ≈ expected_pressure
		@test new_state.elastic_state.s == zeros(SMatrix{3, 3, Float64, 9})
		@test get_soundspeed(material, new_state) ≈ expected_soundspeed
	end

	@testset "Deviatoric Deformation" begin
		velocity_gradient = @SMatrix [1.0 0.0 0.0;
									  0.0 -1.0 0.0;
									  0.0 0.0 0.0]
		dt = 0.1
		stress, new_state = material_model(material, state, identity, velocity_gradient, 1.0, 1.0, dt, 0.01)
		expected_deviatoric_stress = 2.0 * elasticity.μ * dt * velocity_gradient

		@test stress ≈ expected_deviatoric_stress
		@test new_state.elastic_state.s ≈ expected_deviatoric_stress
		@test new_state.eos_state.p == 0.0
	end
end
