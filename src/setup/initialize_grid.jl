function autogrid_parameters!(positions, ppc_1d=2; cells_to_border=4, padding=2, max_dx, min_dx)
    @assert ppc_1d > 0 "Particles per cell (ppc_1d) must be greater than 0"
    if max_dx !== nothing
        @assert max_dx > 0 "max_dx must be greater than 0"
    end
    if min_dx !== nothing
        @assert min_dx > 0 "min_dx must be greater than 0"
    end
    if max_dx !== nothing && min_dx !== nothing
        @assert max_dx >= min_dx "max_dx must be greater than or equal to min_dx"
    end

    # Find dx
 
    dx_particles = median_nn_spacing(positions)
    dx = dx_particles * ppc_1d

    if max_dx !== nothing
        dx = min(dx, max_dx)
    end
    if min_dx !== nothing
        dx = max(dx, min_dx)
    end

    # Compute the bounding box of the particle positions
    origin = minimum(positions, dims=1) .- cells_to_border * dx
    max_pos = maximum(positions, dims=1) .+ cells_to_border * dx

    Nx = ceil(Int, (max_pos[1] - origin[1]) * inv_dx) + 2 * padding
    Ny = ceil(Int, (max_pos[2] - origin[2]) * inv_dx) + 2 * padding
    Nz = ceil(Int, (max_pos[3] - origin[3]) * inv_dx) + 2 * padding
    

    return dx, SVector(Nx, Ny, Nz), SVector(origin...), padding
    
end

function median_nn_spacing(positions::AbstractVector{SVector{3,<:Number}})
    X = hcat(positions...)
    tree = BallTree(X)
    ds, _ = knnquery(tree, X, 2)
    return median(ds[2, :])
end


function initial_p2g_kernel!(grid_state, positions, velocities, masses, origin, inv_dx, padding, spline)
    p_idx = @index(Global, 1)
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
        
        if !checkbounds(grid_state, i, j, k)
            continue
        end

        natural_coords = grid_pos - SVector(i, j, k)
        N = shapefunction(spline, natural_coords)

        @atomic :monotonic grid_state.m[i, j, k] += N * mass
        @atomic :monotonic grid_state.p[i, j, k].x += N * vel[1]
        @atomic :monotonic grid_state.p[i, j, k].y += N * vel[2]
        @atomic :monotonic grid_state.p[i, j, k].z += N * vel[3]
    end
end

function initial_p2g!(grid, positions, velocities, masses, spline)
    grid_old = grid.state_old
    origin = grid.origin
    inv_dx = grid.inv_dx
    padding = grid.padding

    backend = KernelAbstractions.get_backend(grid_old.m)

    kernel = initial_p2g_kernel!(backend)

    kernel(grid_old, positions, velocities, masses, origin, inv_dx, padding, spline;
            ndrange=length(positions))

    KernelAbstractions.synchronize(backend)
end