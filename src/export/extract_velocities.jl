function extract_velocities(grid, particle_set, spline)

    grid_state = grid.state_old
    T = eltype(grid_state.mass)

    num_particles = length(particle_set.particles.pos)
    velocities = Vector{SVector{3, T}}(undef, num_particles)

    inv_dx = grid.inv_dx
    origin = grid.origin
    padding = grid.padding

    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)
    
    # Performance-Optimierung für die CPU-Schleife
    @inbounds for p_idx in 1:num_particles
        # Dank StructArrays liefert uns der direkte Index-Zugriff ein SVector{3, T}
        pos = particle_set.particles.pos[p_idx]
        
        vel = zero(SVector{3, T})
        
        # Gitter-Koordinaten und Basis-Knoten berechnen
        grid_pos = get_grid_position(pos, inv_dx, origin, padding)
        base_node = get_support_base(spline, grid_pos)
        
        # Loop über die umliegenden Gitterknoten (Support-Domain)
        for di in iterator_i, dj in iterator_j, dk in iterator_k
            i = base_node[1] + di
            j = base_node[2] + dj
            k = base_node[3] + dk
            
            # Boundary-Check für das Gitter
            if !checkbounds(Bool, grid_state.mass, i, j, k)
                continue
            end
            
            # Gewichtung (Shape Function Value) berechnen
            natural_coords = grid_pos - SVector(i, j, k)
            N = shapefunction(spline, natural_coords)
            
            # Nur interpolieren, wenn der Knoten Masse besitzt
            if grid_state.mass[i, j, k] > 0
                # Gitter-Geschwindigkeit berechnen: v = p / m
                # `state.momentum[i, j, k]` liefert direkt einen SVector{3, T}
                v_grid = grid_state.momentum[i, j, k] / grid_state.mass[i, j, k]
                vel += N * v_grid
            end
        end
        
        # Rekonstruierte Geschwindigkeit speichern
        velocities[p_idx] = vel
    end
    
    return velocities

end