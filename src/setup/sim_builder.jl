struct Body{T, S<:AbstractShape, M<:AbstractMaterial} <: AbstractBody
    shape::S
    material::M
end


struct SimulationSetup{T, B, G, P, BC, EF, SF}
    bodies::B
    dx::T
    ppc_1d::Int
    t_max::T
    grid_type::G
    material_group_type::P
    boundary_condition::BC
    external_force::EF
    shapefunction::SF
end

function SimulationSetup(bodies, dx, t_max;
    ppc_1d=2,
    grid_type=DenseGrid,
    material_group_type=SoAMaterialGroup,
    boundary_condition=NoBoundaryCondition(),
    external_force=NoExternalForce(),
    shapefunction=QuadraticSpline()
)
    return SimulationSetup{eltype(dx), typeof(bodies), typeof(grid_type), typeof(material_group_type), typeof(boundary_condition), typeof(external_force), typeof(shapefunction)}(
        bodies, dx, ppc_1d, t_max, grid_type, material_group_type, boundary_condition, external_force, shapefunction
    )    
end


function build_sim(setup::SimulationSetup{T}) where {T}
    # 1. Determine physical spacing between particles
    spacing = setup.dx / setup.ppc_1d
    
    # Pre-allocate containers for tracking global bounds
    all_positions = SVector{3, T}[]
    material_groups = () # Empty tuple to build typestable collections

    # 2. Loop through user-defined bodies and rasterize geometries
    for body in setup.bodies
        # Extract particle data using our optimized shape generators
        pos, vel, mass, vol = generate_particles(
            body.shape, 
            spacing, 
            body.material.ρ, # Pass density from the material model
            zero(SVector{3, T}) # Default linear translation velocity (or expand Body to hold it)
        )
        
        # Track global coordinates for the automatic grid resizing boundary
        append!(all_positions, pos)
        
        # 3. Pack raw data vectors into your static SoA (Struct-of-Arrays) cache structures
        N_particles = length(pos)
        particle_vector = Vector{Particle{T}}(undef, N_particles)
        
        for i in 1:N_particles
            # Assemble individual concrete particle structs
            particle_vector[i] = Particle(
                pos[i], 
                mass[i], 
                vol[i], 
                MaterialCache(body.material) # Initialize material history variables
            )
        end
        
        # Wrap into a SoAMaterialGroup (instantiating your concrete group type)
        # We enforce type stability here by progressively packing the tuple
        concrete_group = setup.material_group_type(particle_vector, body.material)
        material_groups = (material_groups..., concrete_group)
    end
    
    # 4. Automate Grid Sizing based on generated particle extents
    # Reuses your existing 'autogrid_parameters!' function from src/setup/autogrid.jl
    actual_dx, grid_size, origin, padding = autogrid_parameters!(all_positions; max_dx=setup.dx)
    
    # Instantiate the concrete grid (e.g., DenseGrid) based on setup.grid_type
    grid = setup.grid_type(actual_dx, grid_size, origin, padding)
    
    # 5. Assemble and return the rigid, fully parameter-typed core simulation Model
    model = Model(
        material_groups,
        grid,
        setup.boundary_condition,
        setup.external_force,
        setup.shapefunction,
        zero(T),         # Initial time t = 0.0
        setup.t_max
    )
    
    return model
end
