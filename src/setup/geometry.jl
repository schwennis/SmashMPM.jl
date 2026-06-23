# Helper for converting between euler angles and rotation matrices
# Used to orient cylinders and rectangular prisms
function generate_rotation_matrix(roll::T, pitch::T, yaw::T) where T
    R_x = @SMatrix [
        1      0           0      ;
        0  cos(roll)  -sin(roll)  ;
        0  sin(roll)   cos(roll)
    ]

    R_y = @SMatrix [
         cos(pitch)  0  sin(pitch) ;
             0       1      0      ;
        -sin(pitch)  0  cos(pitch)
    ]

    R_z = @SMatrix [
        cos(yaw)  -sin(yaw)  0 ;
        sin(yaw)   cos(yaw)  0 ;
           0          0      1
    ]

    return R_z * R_y * R_x
end


# ---------------------------------------------------------------------------- #
#                              Abstract Shape Type                             #
# ---------------------------------------------------------------------------- #
abstract type AbstractShape end


# ---------------------------------------------------------------------------- #
#                                    Sphere                                    #
# ---------------------------------------------------------------------------- #

struct Sphere{T} <: AbstractShape
    center::SVector{3, T}
    radius::T
end

"""
    generate_particles(shape::Sphere{T}, spacing::T, ρ::T, velocity::SVector{3,T}, [ω_vector::SVector{3,T}]) -> (pos, vel, mass, vol)

Generate a perfectly symmetric grid of particles inside a sphere, centered at `shape.center`.

# Arguments
- `shape::Sphere{T}`: Sphere geometry containing `radius` and `center`.
- `spacing::T`: Initial particle spacing.
- `ρ::T`: Material density.
- `velocity::SVector{3,T}`: Linear velocity vector.
- `ω_vector::SVector{3,T}`: Angular velocity vector for solid-body rotation (default: zeros).

# Returns
A tuple of four aligned vectors:
1. `positions`: Particle coordinates.
2. `velocities`: Particle velocities.
3. `masses`: Particle masses.
4. `volumes`: Particle volumes.
"""
function generate_particles(shape::Sphere{T}, spacing::T, ρ::T, velocity::SVector{3, T}, ω_vector::SVector{3, T}=zero(SVector{3, T})) where T
    R = shape.radius
    center = shape.center
    
    # Number of particles per axis
    num_particles_per_axis = ceil(Int, (2R) / spacing)

    # estimate total number of particles (for size hinting to prevent multiple resizes of the array)
    estimated_particles = round(Int, (4/3 * π * R^3) / (spacing^3))
    
    positions = SVector{3, T}[]
    velocities = SVector{3, T}[]
    sizehint!(positions, estimated_particles)
    sizehint!(velocities, estimated_particles)

    # Offset to center the grid around the origin (0,0,0) before translation to the actual center
    grid_offset = 0.5 * num_particles_per_axis * spacing

    for i in 0:num_particles_per_axis-1
        for j in 0:num_particles_per_axis-1
            for k in 0:num_particles_per_axis-1
                pos = SVector(
                    (i + 0.5) * spacing - grid_offset, 
                    (j + 0.5) * spacing - grid_offset, 
                    (k + 0.5) * spacing - grid_offset
                )
                
                # Check if inside the sphere
                if norm(pos) <= R

                    push!(positions, pos + center)  # Shift to actual center
                    
                    v_rot = !iszero(ω_vector) ? cross(ω_vector, pos) : zero(SVector{3, T})  # Calculate rotational velocity if ω_vector is not zero
                    push!(velocities, velocity + v_rot) # Add linear and rotational velocity
                end
            end
        end
    end

    # Calculate V_0 and m
    N_particles = length(positions)
    V0_scalar = spacing^3
    m_scalar  = ρ * V0_scalar

    volumes = fill(V0_scalar, N_particles)
    masses  = fill(m_scalar, N_particles)

    return positions, velocities, masses, volumes
end



# ---------------------------------------------------------------------------- #
#                                   Cylinder                                   #
# ---------------------------------------------------------------------------- #

struct Cylinder{T} <: AbstractShape
    center::SVector{3, T}
    radius::T
    height::T
    euler_angles::SVector{3, T}  # (roll, pitch, yaw) in radians
end

"""
    generate_particles(shape::Cylinder{T}, spacing::T, ρ::T, velocity::SVector{3,T}, [ω_vector::SVector{3,T}]) -> (pos, vel, mass, vol)

Generate a symmetric grid of particles inside a cylinder, centered at `shape.center`.

# Arguments
- `shape::Cylinder{T}`: Cylinder geometry containing `radius`, `height`, `center` and `euler_angles`.
- `spacing::T`: Initial particle spacing.
- `ρ::T`: Material density.
- `velocity::SVector{3,T}`: Linear velocity vector.
- `ω_vector::SVector{3,T}`: Angular velocity vector for solid-body rotation (default: zeros).

# Returns
A tuple of four aligned vectors:
1. `positions`: Particle coordinates.
2. `velocities`: Particle velocities.
3. `masses`: Particle masses.
4. `volumes`: Particle volumes.
"""
function generate_particles(
    shape::Cylinder{T}, 
    spacing::T, 
    ρ::T, 
    velocity::SVector{3, T}, 
    ω_vector::SVector{3, T}=zero(SVector{3, T})
) where T

    R = shape.radius
    H = shape.height
    center = shape.center
    euler_angles = shape.euler_angles

    # Create Rotation Matrix
    rot_matrix = generate_rotation_matrix(euler_angles[1], euler_angles[2], euler_angles[3])

    # Number of particles per axis
    num_particles_radial = ceil(Int, 2R / spacing)
    num_particles_height = ceil(Int, H / spacing)

    # estimate total number of particles (for size hinting to prevent multiple resizes of the array)
    estimated_particles = round(Int, (π * R^2 * H) / (spacing^3))
    positions = SVector{3, T}[]
    velocities = SVector{3, T}[]
    sizehint!(positions, estimated_particles)
    sizehint!(velocities, estimated_particles)

    # Offset to center the grid around the origin (0,0,0) before translation to the actual center
    grid_offset_radial = 0.5 * num_particles_radial * spacing
    grid_offset_height = 0.5 * num_particles_height * spacing

    for i in 0:num_particles_radial-1
        for j in 0:num_particles_radial-1
            for k in 0:num_particles_height-1
                pos = SVector(
                    (i + 0.5) * spacing - grid_offset_radial, 
                    (j + 0.5) * spacing - grid_offset_radial, 
                    (k + 0.5) * spacing - grid_offset_height
                )
                
                # Check if inside the cylinder (ignore z for radial check)
                if pos[1]^2 + pos[2]^2 <= R^2 && abs(pos[3]) <= H/2
                    pos = rot_matrix * pos  # Apply rotation
                    push!(positions, pos + center)  # Shift to actual center
                    
                    v_rot = !iszero(ω_vector) ? cross(ω_vector, pos) : zero(SVector{3, T})  # Calculate rotational velocity if ω_vector is not zero
                    push!(velocities, velocity + v_rot) # Add linear and rotational velocity
                end
            end
        end
    end

    # Calculate V_0 and m
    N_particles = length(positions)
    V0_scalar = spacing^3
    m_scalar  = ρ * V0_scalar

    volumes = fill(V0_scalar, N_particles)
    masses  = fill(m_scalar, N_particles)

    return positions, velocities, masses, volumes
end


# ---------------------------------------------------------------------------- #
#                              Abstract Body Type                              #
# ---------------------------------------------------------------------------- #
abstract type AbstractBody end

struct Body{S,M,T} <: AbstractBody
    shape::S
    velocity::SVector{3, T}
    rot_vector::SVector{3, T}
    material::M
end