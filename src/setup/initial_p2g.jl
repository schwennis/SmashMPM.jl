@kernel function initial_p2g_kernel!(grid_state, positions, velocities, affine, masses, origin, inv_dx, spline)
    p_idx = @index(Global, Linear)

    pos = positions[p_idx]
    vel = velocities[p_idx]
    mass = masses[p_idx]

    grid_pos = grid_position(pos, inv_dx, origin)
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
        r_rel = - natural_coords * (1 / inv_dx)

        p_update = N * (mass * vel + mass * affine * r_rel) # Stress isnt calculated yet, so we need the factor m
        
        @atomic :monotonic grid_state.mass[i, j, k] += N * mass
        @atomic :monotonic grid_state.momentum.x[i, j, k] += p_update[1]
        @atomic :monotonic grid_state.momentum.y[i, j, k] += p_update[2]
        @atomic :monotonic grid_state.momentum.z[i, j, k] += p_update[3]
    end
end

function initial_p2g!(grid, positions, velocities, affines, masses, spline)
    grid_state = grid.state_write
    origin = grid.origin
    inv_dx = grid.inv_dx

    backend = KernelAbstractions.get_backend(grid_state.mass)

    kernel = initial_p2g_kernel!(backend)

    kernel(grid_state, positions, velocities, affines, masses, origin, inv_dx, spline;
            ndrange=length(positions))

    KernelAbstractions.synchronize(backend)
end


# ---------------------------------------------------------------------------- #
#                           Initial P2G on SparseGrid                          #
# ---------------------------------------------------------------------------- #
# Scatters into grid.state_write, whose tiles must have been activated before
# (activate_tiles!). Afterwards, finalize_initial_p2g! swaps write -> read.
@kernel function initial_p2g_sparse_kernel!(state, map, domain, positions, velocities, affine, masses,
                                            origin, inv_dx, spline)
    p_idx = @index(Global, Linear)

    pos  = positions[p_idx]
    vel  = velocities[p_idx]
    mass = masses[p_idx]

    grid_pos = grid_position(pos, inv_dx, origin)

    if grid_pos_representable(grid_pos)
        base_node = get_support_base(spline, grid_pos)
        iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

        for di in iterator_i, dj in iterator_j, dk in iterator_k
            i = base_node[1] + di
            j = base_node[2] + dj
            k = base_node[3] + dk

            l, t = find_node(map, domain, i, j, k)
            if t == 0
                # outside a BoxDomain: clipped like in the DenseGrid; otherwise a lost write
                node_in_domain(domain, i, j, k) && record_miss!(map)
            else
                natural_coords = grid_pos - SVector(i, j, k)
                N = shapefunction(spline, natural_coords)
                r_rel = - natural_coords * (1 / inv_dx)

                p_update = N * (mass * vel + mass * affine * r_rel)

                @atomic :monotonic state.mass[l, t] += N * mass
                @atomic :monotonic state.momentum.x[l, t] += p_update[1]
                @atomic :monotonic state.momentum.y[l, t] += p_update[2]
                @atomic :monotonic state.momentum.z[l, t] += p_update[3]
            end
        end
    else
        record_miss!(map)
    end
end

function initial_p2g!(grid::SparseGrid, positions, velocities, affines, masses, spline)
    isempty(positions) && return nothing
    state   = grid.state_write
    backend = KernelAbstractions.get_backend(state.mass)
    kernel  = initial_p2g_sparse_kernel!(backend)

    kernel(state, grid.map_write, grid.domain, positions, velocities, affines, masses,
           grid.origin, grid.inv_dx, spline; ndrange=length(positions))

    KernelAbstractions.synchronize(backend)
    return nothing
end


# ---------------------------------------------------------------------------- #
#                       Preparation / finalization per grid                    #
# ---------------------------------------------------------------------------- #
prepare_initial_p2g!(::DenseGrid, particle_sets, spline)  = nothing
finalize_initial_p2g!(grid::DenseGrid, particle_sets, spline) = grid_reset!(grid)   # Swap write <> read

# SparseGrid: activate the write tiles, scatter into them, then swap them to read
# (and activate the next write tiles).
prepare_initial_p2g!(grid::SparseGrid, particle_sets, spline) = (activate_tiles!(grid, particle_sets, spline); nothing)
# The swap also activates the write tiles for the first g2p2g step.
finalize_initial_p2g!(grid::SparseGrid, particle_sets, spline) = grid_reset!(grid, particle_sets, spline)