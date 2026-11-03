# ---------------------------------------------------------------------------- #
#                           G2P2G on the SparseGrid                            #
# ---------------------------------------------------------------------------- #

"""
Presaved tile indices of the (up to 8) tiles that contain the nodes of one stencil.
`t[1 + bx + 2by + 4bz]` is the tile index (0 = tile not active) of the tile
that is `bx, by, bz ∈ {0, 1}` tiles away from the tile of the lowest stencil node.
"""
struct StencilTiles
    t::NTuple{8, Int32}
    ti::Int     # tile coordinates of the lowest stencil node
    tj::Int
    tk::Int
end

@inline function find_stencil_tiles(map::AbstractTileMap, base, lo_off::Int, hi_off::Int)
    ti0, tj0, tk0 = tile_coord(base[1] + lo_off), tile_coord(base[2] + lo_off), tile_coord(base[3] + lo_off)
    ti1, tj1, tk1 = tile_coord(base[1] + hi_off), tile_coord(base[2] + hi_off), tile_coord(base[3] + hi_off)

    tiles = ntuple(Val(8)) do n
        bx = (n - 1) & 1
        by = ((n - 1) >> 1) & 1
        bz = (n - 1) >> 2

        # the upper tile of an axis is only needed if it differs from the lower one
        needed = ((bx == 0) | (ti1 != ti0)) & ((by == 0) | (tj1 != tj0)) & ((bz == 0) | (tk1 != tk0))
        ti = bx == 0 ? ti0 : ti1
        tj = by == 0 ? tj0 : tj1
        tk = bz == 0 ? tk0 : tk1

        if needed && tile_in_key_range(ti, tj, tk)
            Int32(find_tile(map, tile_key(ti, tj, tk)))
        else
            Int32(0)
        end
    end
    return StencilTiles(tiles, ti0, tj0, tk0)
end

"""Tile index of node (i, j, k) from the cached stencil tiles (0 = not active)."""
@inline function node_tile(st::StencilTiles, i, j, k)
    bx = Int(tile_coord(i) != st.ti)
    by = Int(tile_coord(j) != st.tj)
    bz = Int(tile_coord(k) != st.tk)
    return Int(st.t[1 + bx + 2 * by + 4 * bz])
end


# Device function for one particle. Early returns are fine here; the kernel
# below is only a thin wrapper (and AoSoA / shared-memory variants can reuse it).
@inline function g2p2g_sparse_particle!(
    p_idx, particles, material,
    state_read, map_read, state_write, map_write, domain,
    origin, inv_dx::T, spline, dt::T
) where {T}
    mass_cutoff = sqrt(floatmin(T))  # Threshold to avoid division by zero in G2P

    # Extract particle properties
    pos_old = particles.pos[p_idx]
    mass = particles.mass[p_idx]
    V0 = particles.V0[p_idx]

    grid_pos_old = grid_position(pos_old, inv_dx, origin)
    if !grid_pos_representable(grid_pos_old)    # NaN or outside the key range
        record_miss!(map_write)
        return nothing
    end

    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)
    lo_off, hi_off = first(iterator_i), last(iterator_i)

    vel = zero(SVector{3, T})
    B = zero(SMatrix{3, 3, T, 9})

    # ---------------------------------- G2P ---------------------------------- #
    base_node_old = get_support_base(spline, grid_pos_old)
    tiles_old = find_stencil_tiles(map_read, base_node_old, lo_off, hi_off)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node_old[1] + di
        j = base_node_old[2] + dj
        k = base_node_old[3] + dk

        node_in_domain(domain, i, j, k) || continue     # outside of a BoxDomain (OpenDomain: always true)
        t = node_tile(tiles_old, i, j, k)
        t == 0 && continue                              # inactive tile: no mass there
        l = local_index(i, j, k)

        m = state_read.mass[l, t]
        if m > mass_cutoff
            natural_coords = grid_pos_old - SVector(i, j, k)
            r_rel = - natural_coords * (1 / inv_dx)
            N = shapefunction(spline, natural_coords)

            v_grid = state_read.momentum[l, t] / m
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
    wavespeed_new = soundspeed_new + norm(vel)

    affine = - dt * vol_new * σ * M_inv(spline, inv_dx) + mass * C

    # ---------------------------------- P2G ---------------------------------- #
    grid_pos_new = grid_position(pos_new, inv_dx, origin)
    if !grid_pos_representable(grid_pos_new)
        record_miss!(map_write)
        return nothing
    end

    base_node_new = get_support_base(spline, grid_pos_new)
    tiles_new = find_stencil_tiles(map_write, base_node_new, lo_off, hi_off)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node_new[1] + di
        j = base_node_new[2] + dj
        k = base_node_new[3] + dk

        node_in_domain(domain, i, j, k) || continue
        t = node_tile(tiles_new, i, j, k)
        if t == 0
            record_miss!(map_write)     # tile was not activated: particle moved further than ACTIVATION_MARGIN
            continue
        end
        l = local_index(i, j, k)

        natural_coords = grid_pos_new - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx)
        N = shapefunction(spline, natural_coords)

        @atomic :monotonic state_write.mass[l, t] += N * mass
        p_update = N * (mass * vel + affine * r_rel)
        @atomic :monotonic state_write.momentum.x[l, t] += p_update[1]
        @atomic :monotonic state_write.momentum.y[l, t] += p_update[2]
        @atomic :monotonic state_write.momentum.z[l, t] += p_update[3]
        atomic_max!(state_write.wave_speed, l, t, wavespeed_new)
    end

    return nothing
end


@kernel function g2p2g_sparse_kernel!(
    particles, material,
    state_read, map_read, state_write, map_write, domain,
    origin, inv_dx::T, spline, dt::T
) where {T}
    p_idx = @index(Global, Linear)
    g2p2g_sparse_particle!(p_idx, particles, material,
                           state_read, map_read, state_write, map_write, domain,
                           origin, inv_dx, spline, dt)
end


# More specific than g2p2g!(::MPMModel{T}, ::T) for the DenseGrid
function g2p2g!(model::MPMModel{T, MGT, SG}, dt::T) where {T, MGT, SG <: SparseGrid}
    grid = model.grid
    spline = model.shapefunction
    backend = model.backend

    # The tile lookup assumes that a stencil spans at most 2 tiles per axis
    @assert length(get_support_offsets(spline)[1]) <= TILE_SIZE "Shape function support is wider than TILE_SIZE"

    kernel = g2p2g_sparse_kernel!(backend)

    foreach(model.particle_sets) do particle_set
        kernel(particle_set.particles, particle_set.material,
               grid.state_read, grid.map_read, grid.state_write, grid.map_write, grid.domain,
               grid.origin, grid.inv_dx, spline, dt;
               ndrange=length(particle_set.particles))
    end

    KernelAbstractions.synchronize(backend)
    return nothing
end