# ---------------------------------------------------------------------------- #
#                                     G2P2G                                    #
# ---------------------------------------------------------------------------- #
# ONE kernel for every grid type. The kernel only talks to the grid through the
# small "grid access" interface (src/grids/grid_access.jl). Julia compiles it
# separately for every access type, so the dense / sparse decisions are made at
# compile time; nothing is dispatched at run time.
#
#   DenseGrid:  nodes outside the array are clipped, no position check, no miss counter
#   SparseGrid: positions outside the key range / NaN and writes into inactive
#               tiles are counted as misses (reported in grid_reset!)
# ---------------------------------------------------------------------------- #

# Device function for one particle. Early returns are fine here; the kernel
# below is only a thin wrapper (and AoSoA / shared-memory variants can reuse it).
@inline function g2p2g_particle!(
    p_idx, particles, material,
    grid_read, grid_write,
    origin, inv_dx::T, spline, dt::T
) where {T}
    mass_cutoff = sqrt(floatmin(T))  # Threshold to avoid division by zero in G2P

    # Extract particle properties
    pos_old = particles.pos[p_idx]
    mass = particles.mass[p_idx]
    V0 = particles.V0[p_idx]

    grid_pos_old = grid_position(pos_old, inv_dx, origin)
    if !position_ok(grid_read, grid_pos_old)    # SparseGrid: NaN or outside the key range
        record_miss!(grid_write)
        return nothing
    end

    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    vel = zero(SVector{3, T})
    B = zero(SMatrix{3, 3, T, 9})

    # ---------------------------------- G2P ---------------------------------- #
    base_node_old = get_support_base(spline, grid_pos_old)
    cache_old     = stencil_cache(grid_read, spline, base_node_old)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node_old[1] + di
        j = base_node_old[2] + dj
        k = base_node_old[3] + dk

        idx, status = node_slot(grid_read, cache_old, i, j, k)
        status == NODE_OK || continue           # clipped or inactive: no mass there

        m = node_mass(grid_read, idx)
        if m > mass_cutoff
            natural_coords = grid_pos_old - SVector(i, j, k)
            r_rel = - natural_coords * (1 / inv_dx)
            N = shapefunction(spline, natural_coords)

            v_grid = node_momentum(grid_read, idx) / m
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
    σ, mat_state_write = material_model(material, mat_state, F_new, C, V0, mass, dt, 1/inv_dx)
    particles.mat_state[p_idx] = mat_state_write
    J = det(F_new)
    vol_new = J * V0

    soundspeed_new = soundspeed(material, mat_state_write)
    particles.wave_speed[p_idx] = soundspeed_new + norm(vel)


    # ---------------------------------- P2G ---------------------------------- #
    grid_pos_new = grid_position(pos_new, inv_dx, origin)
    if !position_ok(grid_write, grid_pos_new)
        record_miss!(grid_write)
        return nothing
    end

    base_node_new = get_support_base(spline, grid_pos_new)
    cache_new     = stencil_cache(grid_write, spline, base_node_new)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node_new[1] + di
        j = base_node_new[2] + dj
        k = base_node_new[3] + dk

        idx, status = node_slot(grid_write, cache_new, i, j, k)
        if status != NODE_OK
            # tile was not activated: particle moved further than ACTIVATION_MARGIN
            status == NODE_INACTIVE && record_miss!(grid_write)
            continue
        end

        natural_coords = grid_pos_new - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx)
        N = shapefunction(spline, natural_coords)

        p_update = N * mass * (vel + C * r_rel)
        f_update = -N * vol_new * M_inv(spline, inv_dx) * (σ * r_rel)
        scatter!(grid_write, idx, N * mass, p_update, f_update)
    end

    return nothing
end


@kernel function g2p2g_kernel!(
    particles, material,
    grid_read, grid_write,
    origin, inv_dx::T, spline, dt::T
) where {T}
    p_idx = @index(Global, Linear)
    g2p2g_particle!(p_idx, particles, material,
                    grid_read, grid_write,
                    origin, inv_dx, spline, dt)
end


"""
    g2p2g!(model, dt)

G2P from `grid.state_read`, particle update, P2G into `grid.state_write`.
Works for every grid type that implements the grid access interface.
"""
function g2p2g!(model::MPMModel{T}, dt::T) where {T}
    grid    = model.grid
    spline  = model.shapefunction
    backend = model.backend

    check_stencil(grid, spline)
    grid_read  = read_access(grid)
    grid_write = write_access(grid)

    kernel = g2p2g_kernel!(backend)

    foreach(model.particle_sets) do particle_set
        kernel(particle_set.particles, particle_set.material,
               grid_read, grid_write,
               grid.origin, grid.inv_dx, spline, dt;
               ndrange=length(particle_set.particles))
    end

    KernelAbstractions.synchronize(backend)
    return nothing
end
