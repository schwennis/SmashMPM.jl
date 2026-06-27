@kwdef struct SimulationSetup{G, P, BC, EF, SF, B, T}
    # Grid
    dx::T
    grid_type::Type{G} = DenseGrid
    padding::Int = 2

    # Properties
    CFL_number::T = 0.4
    t_end::T
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
    # Compute grid Size 
    min_corner, max_corner = bounding_box(all_positions)
    grid_length = max_corner - min_corner
    N = SVector{3, Int}(ceil.(Int, grid_length ./ setup.dx)) .+ 2 * setup.padding
    origin = min_corner .- setup.padding * setup.dx    

    grid = DenseGrid(setup.dx, N, origin, setup.padding, setup.backend)
    
    # Create Particle Sets
    particle_counter = 1
    soundspeeds = Vector{T}()
    particle_sets = map(bodies_data) do data
        mat_cache_type = typeof(get_initial_material_state(data.material))
        
        # Array auf CPU anlegen
        particle_vector = Vector{Particle{T, mat_cache_type}}(undef, length(data.pos))
        @inbounds for i in eachindex(data.pos)
            # Wichtig: initial_material_state frisch generieren (oder deepcopy), 
            # damit nicht alle Partikel denselben Referenz-Speicher teilen!
            particle_vector[i] = Particle(particle_counter, data.pos[i], data.mass[i], data.vol[i], deepcopy(get_initial_material_state(data.material)))
            particle_counter += 1
            push!(soundspeeds, get_soundspeed(data.material, particle_vector[i].mat_state))
        end
        
        # Transfer aufs Backend und Rückgabe als SoA
        return setup.particle_set_type(particle_vector, data.material, setup.backend)
    end

    # Initialize velocities of grid
    for data in bodies_data
        initial_p2g!(grid, data.pos, data.vel, data.mass, soundspeeds, setup.shapefunction)
    end


    return MPMModel(
        particle_sets, 
        grid, 
        setup.boundary_condition, 
        setup.external_force, 
        setup.shapefunction, 
        setup.backend, 
        zero(T), 
        setup.t_end, 
        setup.CFL_number
    )
end