# Interpolate nodal velocities to particles on the CPU
using Base.Threads
function extract_velocities(grid::DenseGrid, particle_set::SoAParticleSet, spline::AbstractShapeFunction)
    grid_state = grid.state_old
    T = eltype(grid_state.mass)

    num_particles = length(particle_set.particles.mass)
    velocities = Vector{SVector{3, T}}(undef, num_particles)
    affines = Vector{SMatrix{3, 3, T, 9}}(undef, num_particles)

    inv_dx = grid.inv_dx
    origin = grid.origin

    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    @threads for p_idx in 1:num_particles
        pos = particle_set.particles.pos[p_idx]
        vel = zero(SVector{3, T})
        B = zero(SMatrix{3, 3, T, 9})

        grid_pos = get_grid_position(pos, inv_dx, origin)
        base_node = get_support_base(spline, grid_pos)

        for di in iterator_i, dj in iterator_j, dk in iterator_k
            i = base_node[1] + di
            j = base_node[2] + dj
            k = base_node[3] + dk

            if !checkbounds(Bool, grid_state.mass, i, j, k)
                continue
            end

            natural_coords = grid_pos - SVector(i, j, k)
            N = shapefunction(spline, natural_coords)
            r_rel = - natural_coords * (1 / inv_dx)

            m_node = grid_state.mass[i, j, k]
            if m_node > sqrt(floatmin(T))  # Avoid division by zero
                v_grid = grid_state.momentum[i, j, k] / m_node
                vel += N * v_grid
                B = B + B_update(spline, N, r_rel, v_grid)
            end
        end
        velocities[p_idx] = vel
        affines[p_idx] = B * M_inv(spline, inv_dx)
    end

    return velocities, affines
end