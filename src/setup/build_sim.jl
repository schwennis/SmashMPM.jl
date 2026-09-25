@kwdef struct SimulationSetup{G, P, BC, EF, SF, B, T}
    # Grid
    dx::T
    grid_type::Type{G} = DenseGrid
    padding::Int = 2

    # Properties
    CFL_number::T = typeof(dx)(0.4)
    t_max::T
    dt_max::T = typeof(dx)(0.01)
    ppc_1d::Int = 2
    particle_set_type::Type{P} = SoAParticleSet
    boundary_condition::BC = NoBoundaryCondition()
    external_force::EF = NoExternalForce()
    shapefunction::SF = QuadraticSpline()

    # Backend
    backend::B = CPU()
end


function bounding_box(positions::AbstractArray{SVector{3, T}}) where {T}
    @assert !isempty(positions) "Positions array is empty. Cannot compute bounding box."
    min_corner = positions[1]
    max_corner = positions[1]

    for pos in positions
        min_corner = min.(min_corner, pos)
        max_corner = max.(max_corner, pos)
    end

    return min_corner, max_corner
end

function build_mpm_model(bodies::Tuple, setup::SimulationSetup{DenseGrid, P, BC, EF, SF, B, T}) where {T, P<:AbstractParticleSet, BC<:AbstractBoundaryCondition, EF<:AbstractExternalForce, SF<:AbstractShapeFunction, B}
    particle_spacing = setup.dx / setup.ppc_1d

    all_positions = SVector{3, T}[]

    bodies_data = map(bodies) do body
        pos, vel, mass, vol = generate_particles(body.shape, particle_spacing, body.material.ρ, body.velocity, body.rot_vector)
        return (pos=pos, vel=vel, mass=mass, vol=vol, material=body.material)
    end

    # Positionen für die Bounding Box sammeln
    for data in bodies_data
        append!(all_positions, data.pos)
    end

    # Create Grid
    min_corner, max_corner = bounding_box(all_positions)
    grid_length = max_corner - min_corner .+ particle_spacing
    N = SVector{3, Int}(ceil.(Int, grid_length ./ setup.dx)) .+ 2 * setup.padding
    origin = min_corner .- (setup.padding) * setup.dx .- T(0.5) * particle_spacing

    grid = DenseGrid(setup.dx, N, origin, setup.padding, setup.backend)
    
    # Create Particle Sets
    particle_counter = 1
    particle_sets = map(bodies_data) do data
        mat_state_type = typeof(get_initial_material_state(data.material))
        n_p = length(data.pos)
        
        # Lokale Arrays pro Body auf CPU
        particle_vector = Vector{Particle{T, mat_state_type}}(undef, n_p)
        soundspeeds_cpu = Vector{T}(undef, n_p)

        @inbounds for i in 1:n_p
            state_i = get_initial_material_state(data.material)
            particle_vector[i] = Particle(particle_counter, data.pos[i], data.mass[i], data.vol[i], state_i)
            soundspeeds_cpu[i] = get_soundspeed(data.material, state_i)
            particle_counter += 1
        end

        # Daten für initial_p2g! auf Backend laden
        pos_dev         = _to_backend(setup.backend, data.pos)
        vel_dev         = _to_backend(setup.backend, data.vel)
        mass_dev        = _to_backend(setup.backend, data.mass)
        soundspeeds_dev = _to_backend(setup.backend, soundspeeds_cpu)

        initial_p2g!(grid, pos_dev, vel_dev, mass_dev, soundspeeds_dev, setup.shapefunction)

        # Return ParticleSet auf dem Backend
        return setup.particle_set_type(particle_vector, data.material, setup.backend)
    end
    
    return MPMModel(
        particle_sets, 
        grid, 
        setup.boundary_condition, 
        setup.external_force, 
        setup.shapefunction, 
        setup.backend,
        zero(T), 
        setup.t_max, 
        setup.dt_max,
        setup.CFL_number
    )
end

function max_wavespeed(grid::DenseGrid)
    return maximum(grid.state_old.wave_speed)
end