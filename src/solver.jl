# ---------------------------------------------------------------------------- #
#                               Courant Timestep                               #
# ---------------------------------------------------------------------------- #
function courant_timestep(model::Model, cfl_factor::T=0.5) where {T} 
    # Extract necessary information from the model
    grid = model.grid
   
    dt_courant = cfl_factor / (grid.inv_dx * max_wavespeed(grid))
    
    return dt_courant
end



# ---------------------------------------------------------------------------- #
#                                     G2P2G                                    #
# ---------------------------------------------------------------------------- #
@kernel function g2p2g_kernel!(
    state_old, state_new,
    particle_set,
    origin, inv_dx::T, padding, spline,
    dt::T
) where T
    p_idx = @index(Global, Linear)

    if p_idx > length(particle_set.particles.pos)
        return
    end

    # Extract particle properties
    pos_old = particle_set.particles.pos[p_idx]
    mass = particle_set.particles.mass[p_idx]
    V0 = particle_set.particles.initial_volume[p_idx]

    vel = zero(SVector{3, T})
    C = zero(SMatrix{3, 3, T, 9})

    grid_pos_old = get_grid_position(pos_old, inv_dx, origin, padding)
    base_node_old = get_support_base(spline, grid_pos_old)
    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node_old[1] + di
        j = base_node_old[2] + dj
        k = base_node_old[3] + dk
        
        if !checkbounds(Bool, state_old.mass, i, j, k)
            continue
        end

        natural_coords = grid_pos_old - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx) 
        N = shapefunction(spline, natural_coords)

        # G2P: Interpolate grid velocity to particle
        if state_old.mass[i, j, k] > 0
            v_grid = state_old.momentum[i, j, k] / state_old.mass[i, j, k]
            vel = vel + N * v_grid
            C = C + B_update(spline, N, r_rel, v_grid)
        end
    end

    # Update particle velocity and position
    particle_set.particles.pos[p_idx] = pos_old + vel * dt
    
    # Finalize affine velocity update
    C = C * M_inv(spline, inv_dx)

    # Update particle state
    σ, mat_state_new = update_material_state(particle_set.particles.mat_state[p_idx], C, dt, V0)
    particle_set.particles.mat_state[p_idx] = mat_state_new
    particle_set.particles.F[p_idx] = (I + C * dt) * particle_set.particles.F[p_idx]
    J = det(particle_set.particles.F[p_idx])
    vol_new = J * V0


    soundspeed_new = get_soundspeed(particle_set.material, mat_state_new)
    wavespeed_new = soundspeed_new + norm(vel)
    
    affine = - dt * vol_new * σ * M_inv(spline, inv_dx) + mass * C

    # P2G: Transfer updated particle state to state_new
    grid_pos_new = get_grid_position(particle_set.particles.pos[p_idx], inv_dx, origin, padding)
    base_node_new = get_support_base(spline, grid_pos_new)
    # iterator_i, iterator_j, iterator_k = get_support_offsets(spline)
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

function g2p2g!(model::MPMModel, dt)
    grid = model.grid
    particle_sets = model.particle_sets
    spline = model.shapefunction

    origin = grid.origin
    inv_dx = grid.inv_dx
    padding = grid.padding

    backend = KernelAbstractions.get_backend(grid.state_old.mass)

    kernel = g2p2g_kernel!(backend)

    for particle_set in particle_sets
        kernel(grid.state_old, grid.state_new, particle_set, origin, inv_dx, padding, spline, dt;
                ndrange=length(particle_set.particles.pos))
    end

    KernelAbstractions.synchronize(backend)
    
end