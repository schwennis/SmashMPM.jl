include("../src/SmashMPM.jl")
using .SmashMPM
using LinearAlgebra
using StaticArrays
using StructArrays
using BenchmarkTools


function create_sphere(radius::T, mass::T, num_particles_per_dim, angular_velocity=0) where T <: Real
    # Create a grid of points in a cube that bounds the sphere
    x = range(-radius, radius, length=num_particles_per_dim)
    y = range(-radius, radius, length=num_particles_per_dim)
    z = range(-radius, radius, length=num_particles_per_dim)

    dx = x[2] - x[1]
    V = dx^3
    
    
    positions = SVector{3, T}[]
    velocities = SVector{3, T}[]
    
    for xi in x, yi in y, zi in z
        if sqrt(xi^2 + yi^2 + zi^2) <= radius + T(1e-5) 
            push!(positions, SVector(xi, yi, zi))
        end
    end

    mass_per_particle = mass / length(positions)

    for pos in positions
        r = norm(pos)
        if r > 0
            # Compute tangential velocity for solid body rotation
            tangential_direction = SVector(-pos[2], pos[1], 0) / r
            tangential_velocity = angular_velocity * r * tangential_direction
            push!(velocities, tangential_velocity)
        else
            push!(velocities, SVector(0, 0, 0))
        end
    end
    
    return positions, velocities, mass_per_particle, V
end



# ---------------------------------------------------------------------------- #
#                                     Test                                     #
# ---------------------------------------------------------------------------- #
#Particles
R = 1.0
num_particles_per_dim = 20
dx_particles = 2R / (num_particles_per_dim - 1)
angular_velocity = 2π/10 # 0.1 rotation per second

println("Creating sphere with radius $(R), total mass $(1.0) and angular velocity $(angular_velocity) rad/s...")
positions, velocities, m, V = create_sphere(R, 1.0, num_particles_per_dim, angular_velocity) # radius R, num_particles_per_dim particles per dimension, angular_velocity rotation per second

material = NeoHookean(E=1e5, ν=0.3, ρ=m/V)

particle_vector = Vector{Particle{Float64, NoMaterialCache}}(undef, length(positions))
for i in eachindex(positions)
    particle_vector[i] = Particle{Float64, NoMaterialCache}(positions[i], m, V, NoMaterialCache())
end

material_group = SoAMaterialGroup(particle_vector, material)


# Grid
println("Creating grid based on particle positions...")
dx, grid_size, origin, padding = autogrid_parameters!(positions, fixed_particle_distance=dx_particles, ppc_1d=2, cells_to_border=4, padding=0)

println("Particles created: $(length(positions)) with mass $(m), volume $(V) and dx $(dx_particles)")
println("This correspondents to a radius of $(R) and an angular velocity of $(angular_velocity) rad/s")
println("Grid size: $(grid_size), Origin: $(origin), Padding: $(padding), dx: $(dx)")




grid = DenseGrid(dx, grid_size, origin, padding)

model = Model((material_group,), grid, NoBoundaryCondition(), NoExternalForce(), QuadraticSpline(), 0.0, 1.0)

initial_p2g!(model.grid, positions, velocities, fill(m, length(positions)), model.shapefunction)

step!(model)

@benchmark step!(model)
