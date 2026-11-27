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
#
# The same holds for the transfer scheme (src/transfer.jl): APIC has no higher
# modes (all higher-mode values are `nothing`) and without heating the energy
# bookkeeping is a compile-time `false` branch, so APIC() compiles to the plain
# APIC kernel.
# ---------------------------------------------------------------------------- #

# Device function for one particle. Early returns are fine here; the kernel
# below is only a thin wrapper (and AoSoA / shared-memory variants can reuse it).
@inline function g2p2g_particle!(
    p_idx, particles, material,
    grid_read, grid_write,
    origin, inv_dx::T, spline, transfer, dt::T
) where {T}
    mass_cutoff = sqrt(floatmin(T))  # Threshold to avoid division by zero in G2P
    heating = transfer_heating(transfer)
    dx = 1 / inv_dx

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
    b_high = zero_higher(transfer, T)       # projections onto the higher modes (PolyPIC)
    v2_grid = zero(T)                       # Σ w |v_I - u|²  (heating only)
    u_ref = zero(SVector{3, T})             # reference velocity of the projection loss
    have_ref = false
    mix_heat = zero(T)                      # Σ w ΔE_I / m_I  (heating only)

    # ---------------------------------- G2P ---------------------------------- #
    base_node_old = get_support_base(spline, grid_pos_old)
    cache_old     = stencil_cache(grid_read, spline, base_node_old)
    weights_old   = stencil_weights(spline, grid_pos_old, base_node_old)   # once per stencil
    high_old      = higher_stencil(transfer, spline, grid_pos_old)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node_old[1] + di
        j = base_node_old[2] + dj
        k = base_node_old[3] + dk

        idx, status = node_slot(grid_read, cache_old, i, j, k)
        status == NODE_OK || continue           # clipped or inactive: no mass there

        m = node_mass(grid_read, idx)
        if m > mass_cutoff
            natural_coords = grid_pos_old - SVector(i, j, k)
            r_rel = - natural_coords * dx
            N = stencil_weight(weights_old, di, dj, dk)

            v_grid = node_momentum(grid_read, idx) / m
            vel = vel + N * v_grid
            B = B + B_update(spline, N, r_rel, v_grid)
            b_high = accumulate_higher(b_high, N, v_grid, higher_basis(high_old, -natural_coords, (di, dj, dk)))

            if heating
                # Bessel loss relative to the velocity u of the first node (Galilean
                # invariant: ½Σw|v_I|² - ½|v|² = ½Σw|v_I - u|² - ½|v - u|² for Σw = 1).
                # Without u both terms are ~½|v|² and cancel in Float32 at high speed.
                if !have_ref
                    u_ref, have_ref = v_grid, true
                end
                dv = v_grid - u_ref
                v2_grid  += N * dot(dv, dv)
                mix_heat += N * node_e_mix(grid_read, idx) / m   # mass share w m_p / m_I, per unit mass
            end
        end
    end

    # Update particle velocity and position
    pos_new = pos_old + vel * dt
    particles.pos.x[p_idx] = pos_new[1]
    particles.pos.y[p_idx] = pos_new[2]
    particles.pos.z[p_idx] = pos_new[3]
    grid_pos_new = grid_position(pos_new, inv_dx, origin)

    # Finalize affine velocity update
    C = B * M_inv(spline, inv_dx)

    # Higher modes: normalize with the old stencil, regularize the G modes with the
    # norm at the new position (exact minimal rule, see finalize_higher)
    high_new = higher_stencil(transfer, spline, grid_pos_new)
    c_high, e_high, e_damp = finalize_higher(transfer, b_high, high_old, high_new)

    # Update particle state
    F_old = particles.F[p_idx]
    mat_state = particles.mat_state[p_idx]

    if heating
        # Kinetic energy lost by the transfer, per unit mass (all terms ≥ 0):
        #   projection  ½ Σ w |v_I|² - ½ Σ_α ν_α |c_α|², here relative to u_ref:
        #               ½ Σ w |v_I - u|² - ½ |v - u|² - ½ Σ_{α≠0} ν_α |c_α|²
        #   damping     energy of the G modes that the new stencil cannot take back
        #   mixing      share of the P2G mixing loss of the last step
        dvel = vel - u_ref
        e_particle = T(0.5) * (dot(dvel, dvel) + T(0.25) * dx^2 * sum(abs2, C)) + e_high
        heat = T(0.5) * v2_grid - e_particle + e_damp + mix_heat
        mat_state = add_heat(material, mat_state, heat)
    end

    # The material model gets the trial F (needs det F_n = det F_new / det(I + C dt)) and
    # regularizes it internally; the identical regularization (state of t_n) is stored.
    F_new = (one(SMatrix{3, 3, T, 9}) + C * dt) * F_old
    σ, mat_state_write = material_model(material, mat_state, F_new, C, V0, mass, dt, dx)
    F_new = regularize_deformation(material, mat_state, F_new)
    particles.F[p_idx] = F_new
    particles.mat_state[p_idx] = mat_state_write
    J = det(F_new)
    vol_new = J * V0

    soundspeed_new = soundspeed(material, mat_state_write)
    particles.wave_speed[p_idx] = soundspeed_new + norm(vel)


    # ---------------------------------- P2G ---------------------------------- #
    if !position_ok(grid_write, grid_pos_new)
        record_miss!(grid_write)
        return nothing
    end

    base_node_new = get_support_base(spline, grid_pos_new)
    cache_new     = stencil_cache(grid_write, spline, base_node_new)
    weights_new   = stencil_weights(spline, grid_pos_new, base_node_new)
    # heating: node velocities of the read state are the reference of e_mix
    cache_ref     = heating ? stencil_cache(grid_read, spline, base_node_new) : nothing

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
        r_rel = - natural_coords * dx
        N = stencil_weight(weights_new, di, dj, dk)

        # velocity field of the particle at the node; the higher modes use the basis
        # of the NEW stencil (orthogonal to 1 and ρ there: no momentum, no ang. momentum)
        v_node = vel + C * r_rel + eval_higher(c_high, higher_basis(high_new, -natural_coords, (di, dj, dk)), T)

        p_update = N * mass * v_node
        f_update = -N * vol_new * M_inv(spline, inv_dx) * (σ * r_rel)
        scatter!(grid_write, idx, N * mass, p_update, f_update)
        if heating
            # deposited energy relative to the read velocity u_I of the node (same u_I for
            # all particles of the node and in mixing_loss!): Galilean invariant
            dvn = v_node - reference_velocity(grid_read, cache_ref, i, j, k, T)
            scatter_energy!(grid_write, idx, T(0.5) * N * mass * dot(dvn, dvn))
        end
    end

    return nothing
end


@kernel function g2p2g_kernel!(
    particles, material,
    grid_read, grid_write,
    origin, inv_dx::T, spline, transfer, dt::T
) where {T}
    p_idx = @index(Global, Linear)
    g2p2g_particle!(p_idx, particles, material,
                    grid_read, grid_write,
                    origin, inv_dx, spline, transfer, dt)
end



"""
    g2p2g!(model, dt)

G2P from `grid.state_read`, particle update, P2G into `grid.state_write`.
Works for every grid type that implements the grid access interface and for
every transfer scheme (`model.transfer`: APIC or PolyPIC).
"""
function g2p2g!(model::MPMModel{T}, dt::T) where {T}
    grid     = model.grid
    spline   = model.shapefunction
    backend  = model.backend
    transfer = model.transfer

    check_stencil(grid, spline)
    check_transfer(transfer, spline)
    grid_read  = read_access(grid)
    grid_write = write_access(grid)

    kernel = g2p2g_kernel!(backend)

    foreach(model.particle_sets) do particle_set
        kernel(particle_set.particles, particle_set.material,
               grid_read, grid_write,
               grid.origin, grid.inv_dx, spline, transfer, dt;
               ndrange=length(particle_set.particles))
    end

    KernelAbstractions.synchronize(backend)
    return nothing
end
