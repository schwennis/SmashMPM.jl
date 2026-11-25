# Interpolate nodal velocities to particles on the CPU
# One implementation for all grid types via the grid access interface
# (src/grids/grid_access.jl). Nodes in inactive tiles (or outside the grid / a
# BoxDomain) carry no mass and are skipped.
using Base.Threads
function extract_velocities(grid::AbstractGrid, particle_set::SoAParticleSet, spline::AbstractShapeFunction)
    grid_read = read_access(grid)
    T = eltype(grid.state_read.mass)

    num_particles = length(particle_set.particles.mass)
    velocities = Vector{SVector{3, T}}(undef, num_particles)
    affines = Vector{SMatrix{3, 3, T, 9}}(undef, num_particles)

    inv_dx = grid.inv_dx
    origin = grid.origin

    @threads for p_idx in 1:num_particles
        pos = particle_set.particles.pos[p_idx]
        vel, B = _g2p_velocity(grid_read, pos, origin, inv_dx, spline)
        velocities[p_idx] = vel
        affines[p_idx] = B * M_inv(spline, inv_dx)
    end

    return velocities, affines
end

"""G2P of velocity and B matrix at position `pos` (zero if the position cannot be mapped)."""
@inline function _g2p_velocity(grid, pos, origin, inv_dx::T, spline) where {T}
    vel = zero(SVector{3, T})
    B = zero(SMatrix{3, 3, T, 9})

    grid_pos = grid_position(pos, inv_dx, origin)
    position_ok(grid, grid_pos) || return vel, B    # SparseGrid: NaN or outside the key range

    base_node = get_support_base(spline, grid_pos)
    cache     = stencil_cache(grid, spline, base_node)
    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node[1] + di
        j = base_node[2] + dj
        k = base_node[3] + dk

        idx, status = node_slot(grid, cache, i, j, k)
        status == NODE_OK || continue

        natural_coords = grid_pos - SVector(i, j, k)
        N = shapefunction(spline, natural_coords)
        r_rel = - natural_coords * (1 / inv_dx)

        m_node = node_mass(grid, idx)
        if m_node > sqrt(floatmin(T))  # Avoid division by zero
            v_grid = node_momentum(grid, idx) / m_node
            vel += N * v_grid
            B = B + B_update(spline, N, r_rel, v_grid)
        end
    end
    return vel, B
end
