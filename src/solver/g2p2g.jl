
# ---------------------------------------------------------------------------- #
#                                     G2P2G                                    #
# ---------------------------------------------------------------------------- #
@kernel function g2p2g_kernel!(
    state_old, state_new,
    particles, material,
    origin, inv_dx::T, spline,
    dt::T
) where T
    mass_cutoff = sqrt(floatmin(T))  # Threshold to avoid division by zero in G2P

    p_idx = @index(Global, Linear)

    # Extract particle properties
    pos_old = particles.pos[p_idx]
    mass = particles.mass[p_idx]
    V0 = particles.initial_volume[p_idx]

    vel = zero(SVector{3, T})
    B = zero(SMatrix{3, 3, T, 9})

    grid_pos_old = get_grid_position(pos_old, inv_dx, origin)
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
        if state_old.mass[i, j, k] > mass_cutoff
            v_grid = state_old.momentum[i, j, k] / state_old.mass[i, j, k]
            vel = vel + N * v_grid
            B = B + B_update(spline, N, r_rel, v_grid)
        end
    end

    # Update particle velocity and position
    pos_new = pos_old + vel * dt
    particles.pos.x[p_idx] = pos_new[1]
    particles.pos.y[p_idx] = pos_new[2]
    particles.pos.z[p_idx] = pos_new[3]
    
    # Finalize affine velocity update
    C = B * M_inv(spline, inv_dx)

    # Update particle state
    F_old = particles.F[p_idx]
    mat_state = particles.mat_state[p_idx]

    F_new = (one(SMatrix{3, 3, T, 9}) + C * dt) * F_old
    particles.F[p_idx] = F_new
    σ, mat_state_new = material_model(material, mat_state, F_new, C, V0, mass, dt, 1/inv_dx)
    particles.mat_state[p_idx] = mat_state_new
    J = det(F_new)
    vol_new = J * V0


    soundspeed_new = get_soundspeed(material, mat_state_new)
    wavespeed_new = soundspeed_new + norm(vel)
    
    affine = - dt * vol_new * σ * M_inv(spline, inv_dx) + mass * C

    # P2G: Transfer updated particle state to state_new
    grid_pos_new = get_grid_position(pos_new, inv_dx, origin)
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
        @atomic :monotonic state_new.wave_speed[i, j, k] max wavespeed_new
    end

end
 
function g2p2g!(model::MPMModel{T}, dt::T) where T
    grid = model.grid
    particle_sets = model.particle_sets
    spline = model.shapefunction

    origin = grid.origin
    inv_dx = grid.inv_dx

    backend = model.backend

    kernel = g2p2g_kernel!(backend)

    foreach(particle_sets) do particle_set
        kernel(grid.state_old, grid.state_new, particle_set.particles, particle_set.material, origin, inv_dx, spline, dt;
               ndrange=length(particle_set.particles))
    end

    KernelAbstractions.synchronize(backend)
end