@kernel function initial_p2g_kernel!(grid_state, positions, velocities, masses, soundspeeds, origin, inv_dx, padding, spline)
    p_idx = @index(Global, Linear)
    if p_idx > length(positions)
        error("Particle index out of bounds in initial_p2g_kernel!")
    end

    pos = positions[p_idx]
    vel = velocities[p_idx]
    mass = masses[p_idx]

    grid_pos = get_grid_position(pos, inv_dx, origin, padding)
    base_node = get_support_base(spline, grid_pos)
    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node[1] + di
        j = base_node[2] + dj
        k = base_node[3] + dk
        
        if !checkbounds(Bool, grid_state, i, j, k)
            continue
        end

        natural_coords = grid_pos - SVector(i, j, k)
        N = shapefunction(spline, natural_coords)
        
        @atomic :monotonic grid_state.mass[i, j, k] += N * mass
        @atomic :monotonic grid_state.momentum.x[i, j, k] += N * vel[1]
        @atomic :monotonic grid_state.momentum.y[i, j, k] += N * vel[2]
        @atomic :monotonic grid_state.momentum.z[i, j, k] += N * vel[3]
        @atomic grid_state.wave_speed[i, j, k] = max(grid_state.wave_speed[i, j, k], soundspeeds[p_idx] + norm(vel))
    end
end

function initial_p2g!(grid, positions, velocities, masses, soundspeeds, spline)
    grid_old = grid.state_old
    origin = grid.origin
    inv_dx = grid.inv_dx
    padding = grid.padding

    backend = KernelAbstractions.get_backend(grid_old.mass)

    kernel = initial_p2g_kernel!(backend)

    kernel(grid_old, positions, velocities, masses, soundspeeds, origin, inv_dx, padding, spline;
            ndrange=length(positions))

    KernelAbstractions.synchronize(backend)
end