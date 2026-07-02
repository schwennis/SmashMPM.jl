using SmashMPM
using StaticArrays
using BenchmarkTools
using Profile
using LinearAlgebra
using Atomix: @atomic
using AMDGPU
using KernelAbstractions

# backend = ROCBackend()
backend = CPU()


center = SVector{3, Float64}(0.0, 0.0, 0.0)
R = 1.0
h = 1.0
euler_angles = SVector{3, Float64}(0.0, 0.0, 0.0)

shape1 = Cylinder(center, R, h, euler_angles)

linear_vel = SVector{3, Float64}(0.0, 0.0, 0.0)
rot_speed = 2π/5.0 
rotational_vec = SVector{3, Float64}(0.0, 0.0, rot_speed)
mat = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)

body = Body(shape1, linear_vel, rotational_vec, mat)
bodies = (body,)

Setup = SimulationSetup(dx=0.1, t_end=1.0, backend=backend)

model = build_mpm_model(bodies, Setup)

N_particles = length(model.particle_sets[1].particles.pos)
dims = size(model.grid.state_old.mass)

println("Model built successfully with $N_particles particles and a grid of size $dims.")

function g2p2g_non_kernel(model::MPMModel, dt)
    grid = model.grid
    particle_sets = model.particle_sets
    spline = model.shapefunction

    origin = grid.origin
    inv_dx = grid.inv_dx
    padding = grid.padding

    for particle_set in particle_setss
        for p_idx in 1:length(particle_set.particles.pos)
            g2p2g_quasi_kernel!(grid.state_old, grid.state_new, particle_set, origin, inv_dx, padding, spline, dt, p_idx)
        end
    end

end

function g2p2g_quasi_kernel!(
    state_old, state_new,
    particle_set,
    origin, inv_dx::T, padding, spline,
    dt::T, p_idx::Int
) where T
    # p_idx = @index(Global, Linear)

    # Extract particle properties
    pos_old = particle_set.particles.pos[p_idx]
    mass = particle_set.particles.mass[p_idx]
    V0 = particle_set.particles.initial_volume[p_idx]

    vel = zero(SVector{3, T})
    B = zero(SMatrix{3, 3, T, 9})

    grid_pos_old = SmashMPM.get_grid_position(pos_old, inv_dx, origin, padding)
    base_node_old = SmashMPM.get_support_base(spline, grid_pos_old)
    iterator_i, iterator_j, iterator_k = SmashMPM.get_support_offsets(spline)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node_old[1] + di
        j = base_node_old[2] + dj
        k = base_node_old[3] + dk
        
        if !checkbounds(Bool, state_old.mass, i, j, k)
            continue
        end

        natural_coords = grid_pos_old - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx)
        # @assert type(r_rel) == SVector{3, T} "r_rel should be of type SVector{3, T}, but got $(typeof(r_rel))"
        N = shapefunction(spline, natural_coords)

        # G2P: Interpolate grid velocity to particle
        if state_old.mass[i, j, k] > 0
            v_grid = state_old.momentum[i, j, k] / state_old.mass[i, j, k]
            vel = vel + N * v_grid
            B = B + SmashMPM.B_update(spline, N, r_rel, v_grid)
        end
    end

    # Update particle velocity and position
    particle_set.particles.pos[p_idx] = pos_old + vel * dt
    
    # Finalize affine velocity update
    C = B * SmashMPM.M_inv(spline, inv_dx)

    # Update particle state
    F_old = particle_set.particles.F[p_idx]
    material = particle_set.material
    mat_state = particle_set.particles.mat_state[p_idx]

    σ, mat_state_new = material_model(material, mat_state, F_old, C, V0, mass, dt)
    particle_set.particles.mat_state[p_idx] = mat_state_new
    particle_set.particles.F[p_idx] = (I + C * dt) * particle_set.particles.F[p_idx]
    J = det(particle_set.particles.F[p_idx])
    vol_new = J * V0


    soundspeed_new = get_soundspeed(particle_set.material, mat_state_new)
    wavespeed_new = soundspeed_new + norm(vel)
    
    affine = - dt * vol_new * σ * SmashMPM.M_inv(spline, inv_dx) + mass * C

    # P2G: Transfer updated particle state to state_new
    grid_pos_new = SmashMPM.get_grid_position(particle_set.particles.pos[p_idx], inv_dx, origin, padding)
    base_node_new = SmashMPM.get_support_base(spline, grid_pos_new)
    # iterator_i, iterator_j, iterator_k = SmashMPM.get_support_offsets(spline)
    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node_new[1] + di
        j = base_node_new[2] + dj
        k = base_node_new[3] + dk
        
        if !checkbounds(Bool, state_new.mass, i, j, k)
            continue
        end

        natural_coords = grid_pos_new - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx)
        N = shapefunction(spline, natural_coords)

        @atomic :monotonic state_new.mass[i, j, k] += N * mass
        p_update = N * (mass * vel + affine * r_rel)
        @atomic :monotonic state_new.momentum.x[i, j, k] += p_update[1]
        @atomic :monotonic state_new.momentum.y[i, j, k] += p_update[2]
        @atomic :monotonic state_new.momentum.z[i, j, k] += p_update[3]
        @atomic :monotonic state_new.wave_speed[i, j, k] = max(state_new.wave_speed[i, j, k], wavespeed_new)
    end

end



g2p2g!(model, 1e-5)
display(@benchmark g2p2g!($model, 1e-5))

Profile.clear_malloc_data()

g2p2g!(model, 1e-5)