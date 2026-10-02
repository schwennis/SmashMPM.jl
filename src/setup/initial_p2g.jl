using Base.Threads

function estimate_velocity_gradients(positions, velocities, particle_spacing::T, R::T=T(2)*particle_spacing) where T
    cell(x) = (floor(Int, x[1]/R), floor(Int, x[2]/R), floor(Int, x[3]/R))
    
    cells = Dict{Tuple{Int, Int, Int}, Vector{Int}}()
    for (i, x) in enumerate(positions)
        push!(get!(cells, cell(x), Int[]), i)
    end

    C = Vector{SMatrix{3,3,T,9}}(undef, length(positions))

    @threads for p in eachindex(positions)
        xp, vp = positions[p], velocities[p]
        cx, cy, cz = cell(xp)

        A = zero(SMatrix{3,3,T,9})     # Σ w (Δv)(Δx)ᵀ
        M = zero(SMatrix{3,3,T,9})     # Σ w (Δx)(Δx)ᵀ

        for dx in -1:1, dy in -1:1, dz in -1:1
            list = get(cells, (cx+dx, cy+dy, cz+dz), nothing)
            list === nothing && continue
            for q in list
                q == p && continue
                r  = positions[q] - xp
                d2 = dot(r, r)
                d2 >= R^2 && continue
                w  = (one(T) - d2 / R^2)^2          # glatte, kompakte Gewichtung
                A += w * (velocities[q] - vp) * r'
                M += w * (r * r')
            end
        end

        # Rang prüfen (isolierte Partikel): dann C = 0
        if det(M) > T(1e-6) * (tr(M) / 3)^3
            C[p] = A / M                            # A * inv(M)
        else
            C[p] = zero(SMatrix{3,3,T,9})
        end
    end
    return C

end



@kernel function initial_p2g_kernel!(grid_state, positions, velocities, affines, masses, soundspeeds, origin, inv_dx, spline)
    p_idx = @index(Global, Linear)

    pos = positions[p_idx]
    vel = velocities[p_idx]
    affine = affines[p_idx]
    mass = masses[p_idx]

    grid_pos = get_grid_position(pos, inv_dx, origin)
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
        @atomic :monotonic grid_state.wave_speed[i, j, k] = max(grid_state.wave_speed[i, j, k], soundspeeds[p_idx] + norm(vel))
    end
end

function initial_p2g!(grid, positions, velocities, affines, masses, soundspeeds, spline)
    grid_old = grid.state_old
    origin = grid.origin
    inv_dx = grid.inv_dx

    backend = KernelAbstractions.get_backend(grid_old.mass)

    kernel = initial_p2g_kernel!(backend)

    kernel(grid_old, positions, velocities, affines, masses, soundspeeds, origin, inv_dx, spline;
            ndrange=length(positions))

    KernelAbstractions.synchronize(backend)
end