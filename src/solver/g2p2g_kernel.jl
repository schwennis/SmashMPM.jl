@kernel function g2p2g_kernel!(grid_old, grid_new, origin, inv_dx, padding,
                                particles, material::MatType, spline::SF, dt::T) where {T, MatType<:AbstractMaterial, SF<:AbstractShapeFunction}
    p_idx = @index(Global)

    if p_idx > length(particles)
        error("Particle index out of bounds in g2p2g_kernel!")
    end

    mp = particles[p_idx]


    # =========================================================
    # G2P Phase
    # =========================================================
    v_p = zero(SVector{3, T})
    B_p = zero(SMatrix{3, 3, T, 9})

    grid_pos = get_grid_position(mp.pos, inv_dx, origin, padding)

    base_node = get_support_base(spline, grid_pos)
    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)


    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node[1] + di
        j = base_node[2] + dj
        k = base_node[3] + dk
        
        if !checkbounds(grid_old, i, j, k) || grid_old.m[i, j, k] <= eps(T)
            continue
        end

        natural_coords = grid_pos - SVector(i, j, k)
        r_rel = - natural_coords ./ inv_dx

        N = shapefunction(spline, natural_coords)

        v_I_new = grid_old.p[i, j, k] ./ grid_old.m[i, j, k]

        v_p = v_p + N * v_I_new

        B_p = B_p + B_update(spline, N, r_rel, v_I_new)
    end

    C_p = B_p * M_inv(QuadraticSpline(), inv_dx)

    # =========================================================
    # Particle Update Phase
    # =========================================================
    mp.pos = mp.pos + v_p * dt
    mp.F = (I + C_p * dt) * mp.F
    J_p = det(mp.F)

    σ_p, mp.mat_cache = material_model(material, mp.mat_cache, mp.F, C_p, mp.V0, mp.m, dt)


    soundspeed_p = get_soundspeed(material, mp.mat_cache)
    
    # =========================================================
    # P2G Phase
    # =========================================================

    grid_pos = get_grid_position(mp.pos, inv_dx, origin, padding)

    base_node = get_support_base(spline, grid_pos)
    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node[1] + di
        j = base_node[2] + dj
        k = base_node[3] + dk
        
        if !checkbounds(grid_new, i, j, k)
            continue
        end

        natural_coords = grid_pos - SVector(i, j, k)
        r_rel = - natural_coords ./ inv_dx

        N = shapefunction(spline, natural_coords)
        Q = dt * J_p * mp.V0 * σ_p * M_inv(QuadraticSpline(), inv_dx) + mp.m * C_p
        p_update = N * Q * r_rel

        @atomic :monotonic grid_new.m[i, j, k] += N * mp.m
        @atomic grid_new.wave_speed[i, j, k] = max(grid_new.wave_speed[i, j, k], soundspeed_p + sqrt(sum(v_p.^2)))

        @atomic :monotonic grid_new.p[i, j, k].x += p_update[1]
        @atomic :monotonic grid_new.p[i, j, k].y += p_update[2]
        @atomic :monotonic grid_new.p[i, j, k].z += p_update[3]
    end

end


function g2p2g!(grid::DenseGrid{T, S}, material_group, spline, dt) where {T, S}
    grid_old = grid.state_old
    grid_new = grid.state_new
    origin = grid.origin
    inv_dx = grid.inv_dx
    padding = grid.padding

    particles = material_group.particles
    material = material_group.material

    backend = KernelAbstractions.get_backend(grid_old.m)

    kernel = g2p2g_kernel!(backend)

    kernel(grid_old, grid_new, origin, inv_dx, padding,
            particles, material, spline, dt;
            ndrange=length(particles))

    KernelAbstractions.synchronize(backend)
end

