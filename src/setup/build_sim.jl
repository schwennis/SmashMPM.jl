# ---------------------------------------------------------------------------- #
#                      Domain options for the SparseGrid                       #
# ---------------------------------------------------------------------------- #
# SimulationSetup(domain = ...) accepts
#   AutoDomain()   (default) open domain for NullBoundaryCondition, otherwise a
#                  box fitted around the bodies (like the DenseGrid)
#   OpenDomain()   always unbounded (open boundary conditions)
#   FittedBox()    always a box fitted around the bodies incl. padding + buffer_width
# The DenseGrid ignores this option (it is always a fitted box).
struct AutoDomain end
struct FittedBox end

resolve_domain(::AutoDomain, ::NullBoundaryCondition, N) = OpenDomain()
resolve_domain(::AutoDomain, ::AbstractBoundaryCondition, N) = BoxDomain(N)
resolve_domain(::FittedBox, ::AbstractBoundaryCondition, N) = BoxDomain(N)
resolve_domain(::OpenDomain, ::NullBoundaryCondition, N) = OpenDomain()
resolve_domain(::OpenDomain, bc::AbstractBoundaryCondition, N) =
    error("$(typeof(bc)) needs walls and cannot be combined with an OpenDomain. " *
          "Use NullBoundaryCondition for open boundaries or domain = FittedBox().")


@kwdef struct SimulationSetup{G, P, BC, EF, SF, B, T, E, DM}
    # Grid
    dx::T
    grid_type::Type{G} = DenseGrid
    padding::Int = 2
    buffer_width::SVector{3, Int} = zero(SVector{3, Int})
    domain::DM = AutoDomain()           # only used by the SparseGrid

    # Properties
    cfl_number::T = typeof(dx)(0.4)
    t_max::T
    dt_max::T = typeof(dx)(0.01)
    ppc_1d::Int = 2
    particle_set_type::Type{P} = SoAParticleSet
    boundary_condition::BC = NullBoundaryCondition()
    external_force::EF = NoExternalForce()
    shapefunction::SF = QuadraticSpline()

    # Export Settings
    exporter::E = NoExporter()
    export_time_interval::T = typeof(dx)(0.1)

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

# ---------------------------------------------------------------------------- #
#                               Grid construction                              #
# ---------------------------------------------------------------------------- #
function _build_grid(::Type{<:DenseGrid}, setup::SimulationSetup, N, origin, positions)
    return DenseGrid(setup.dx, N, origin, setup.padding, setup.backend)
end

function _build_grid(::Type{<:SparseGrid}, setup::SimulationSetup, N, origin, positions)
    domain    = resolve_domain(setup.domain, setup.boundary_condition, N)
    n_tiles   = count_tiles(positions, setup.dx, origin, setup.shapefunction; domain)
    max_tiles = max(64, cld(5 * n_tiles, 4))        # 25 % headroom, the pool grows if needed
    return SparseGrid(setup.dx, origin; domain, padding=setup.padding,
                      backend=setup.backend, max_tiles)
end


# ---------------------------------------------------------------------------- #
#                                 Model builder                                #
# ---------------------------------------------------------------------------- #
function build_mpm_model(bodies::Tuple, setup::SimulationSetup{G, P, BC, EF, SF, B, T}) where {G, T, P<:AbstractParticleSet, BC<:AbstractBoundaryCondition, EF<:AbstractExternalForce, SF<:AbstractShapeFunction, B}
    particle_spacing = setup.dx / setup.ppc_1d

    all_positions = SVector{3, T}[]

    bodies_data = map(bodies) do body
        pos, vel, mass, vol = generate_particles(body.shape, particle_spacing, body.material.ρ, body.velocity, body.rot_vector)
        return (pos=pos, vel=vel, mass=mass, vol=vol, affine=initial_affine_matrix(body), material=body.material)
    end

    # Positionen für die Bounding Box sammeln
    for data in bodies_data
        append!(all_positions, data.pos)
    end

    # Grid geometry (for an OpenDomain only the origin matters)
    min_corner, max_corner = bounding_box(all_positions)
    grid_length = max_corner - min_corner .+ particle_spacing

    margin = setup.padding .+ setup.buffer_width
    
    N = SVector{3, Int}(ceil.(Int, grid_length ./ setup.dx)) .+ 2 * margin
    origin = min_corner .- margin * setup.dx .- T(0.5) * particle_spacing

    grid = _build_grid(setup.grid_type, setup, N, origin, all_positions)
    
    # Create Particle Sets (+ device copies of the data needed for the initial P2G)
    particle_counter = 1
    built = map(bodies_data) do data
        mat_state_type = typeof(initial_material_state(data.material))
        n_p = length(data.pos)
        
        # Lokale Arrays pro Body auf CPU
        particle_vector = Vector{Particle{T, mat_state_type}}(undef, n_p)
        soundspeeds_cpu = Vector{T}(undef, n_p)

        @inbounds for i in 1:n_p
            state_i = initial_material_state(data.material)
            particle_vector[i] = Particle(particle_counter, data.pos[i], data.mass[i], data.vol[i], state_i)
            soundspeeds_cpu[i] = soundspeed(data.material, state_i)
            particle_counter += 1
        end

        particle_set = setup.particle_set_type(particle_vector, data.material, setup.backend)

        return (
            set    = particle_set,
            pos    = _to_backend(setup.backend, data.pos),
            vel    = _to_backend(setup.backend, data.vel),
            mass   = _to_backend(setup.backend, data.mass),
            c      = _to_backend(setup.backend, soundspeeds_cpu),
            affine = data.affine,
        )
    end
    particle_sets = map(b -> b.set, built)

    # Initial P2G (SparseGrid: activates its tiles first and swaps them to the read state)
    prepare_initial_p2g!(grid, particle_sets, setup.shapefunction)
    foreach(built) do b
        initial_p2g!(grid, b.pos, b.vel, b.affine, b.mass, b.c, setup.shapefunction)
    end
    finalize_initial_p2g!(grid, particle_sets, setup.shapefunction)
    
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
        setup.cfl_number
    )
end