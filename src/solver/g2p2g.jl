# ---------------------------------------------------------------------------- #
#                                     G2P2G                                    #
# ---------------------------------------------------------------------------- #

# --------------------------- shared helpers --------------------------------- #
"""Skew matrix S = ½[a]× with axial vector ax(S) = a."""
@inline half_cross(a::SVector{3, T}) where {T} =
    SMatrix{3, 3, T, 9}(0, a[3] / 2, -a[2] / 2,
                        -a[3] / 2, 0, a[1] / 2,
                        a[2] / 2, -a[1] / 2, 0)

"""Boundary condition applied to the node velocities v*_i and ṽ_i in the G2P."""
@inline bc_velocity(::NullBoundaryCondition, v, i, j, k, lo, hi, padding) = v
@inline function bc_velocity(::NoSlipBoundary, v, i, j, k, lo, hi, padding)
    wall = (i < lo[1] + padding) | (i > hi[1] - padding) |
           (j < lo[2] + padding) | (j > hi[2] - padding) |
           (k < lo[3] + padding) | (k > hi[3] - padding)
    return wall ? zero(v) : v
end

"""Node range the walls refer to (lo, hi), inclusive, 1-based global indices."""
bc_bounds(::NullBoundaryCondition, grid) = (SVector(1, 1, 1), SVector(1, 1, 1))     # unused
bc_bounds(::NoSlipBoundary, grid::DenseGrid) =
    (SVector(1, 1, 1), SVector{3, Int}(size(grid.state_read.mass)))
bc_bounds(::NoSlipBoundary, grid::SparseGrid{T, S, M, BoxDomain}) where {T, S, M} =
    (grid.domain.lo, grid.domain.hi)
bc_bounds(::NoSlipBoundary, ::SparseGrid{T, S, M, OpenDomain}) where {T, S, M} =
    error("NoSlipBoundary needs walls: use a SparseGrid with a BoxDomain, " *
          "or NullBoundaryCondition for open boundaries.")


# ---------------------------------------------------------------------------- #
#                         Time step of the kick (KDK)                          #
# ---------------------------------------------------------------------------- #
"""
    kick_dt(dt_prev, dt) -> Δt̄

Time-centered kick of the staggered leapfrog, Δt̄_n = (Δt_{n-1} + Δt_n) / 2.
The first step (`dt_prev == 0`, particle velocities at t^0) gets the half kick
Δt_0 / 2 that moves them to t^{1/2}.
"""
@inline kick_dt(dt_prev::T, dt::T) where {T} =
    iszero(dt_prev) ? dt / 2 : (dt_prev + dt) / 2

"""Spin-fix δC = ½[-M⁻¹ Δt (v_adv × v_p)]×: removes Δt m v_adv × v_p from the angular momentum."""
@inline spin_fix(Minv, dt, v_adv, v_p) = half_cross(-Minv * dt * cross(v_adv, v_p))

"""Midpoint exponential exp(A) to 2nd order (A = Δt L^{n+1/2})."""
@inline exp_taylor2(A::SMatrix{3, 3, T, 9}) where {T} = one(A) + A + (A * A) / 2

"""Predicted midpoint x^{n+1/2} = x^n + Δt/2 ṽ^{n-1/2}."""
@inline midpoint_position(particles, p_idx, dt) = particles.pos[p_idx] + (dt / 2) * particles.v_drift[p_idx]


# ---------------------------------------------------------------------------- #
#              Pass 1: kick + AFLIP, P2G into the MUSL buffer at x^{n+1/2}     #
# ---------------------------------------------------------------------------- #
# Early returns are fine here; the kernels below are only thin wrappers.
@inline function g2p2g_pass1_particle!(
    p_idx, particles, scheme,
    grid_read, grid_write, musl,
    bc, lo, hi, padding,
    origin, inv_dx::T, spline, bar_dt::T, dt::T
) where {T}
    mass_cutoff = sqrt(floatmin(T))  # Threshold to avoid division by zero in G2P

    pos  = particles.pos[p_idx]
    mass = particles.mass[p_idx]

    grid_pos = grid_position(pos, inv_dx, origin)
    if !position_ok(grid_read, grid_pos)    # SparseGrid: NaN or outside the key range
        return nothing                      # counted once, in pass 2
    end

    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    v_star  = zero(SVector{3, T});       v_bar_n = zero(SVector{3, T})
    B_star  = zero(SMatrix{3, 3, T, 9}); B_n     = zero(SMatrix{3, 3, T, 9})

    # ------------------------- G2P (kick) at x^n -------------------------- #
    base_node = get_support_base(spline, grid_pos)
    cache     = stencil_cache(grid_read, spline, base_node)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node[1] + di
        j = base_node[2] + dj
        k = base_node[3] + dk

        idx, status = node_slot(grid_read, cache, i, j, k)
        status == NODE_OK || continue           # clipped or inactive: no mass there
        m = node_mass(grid_read, idx)
        m > mass_cutoff || continue

        natural_coords = grid_pos - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx)
        N = shapefunction(spline, natural_coords)

        p_i  = node_momentum(grid_read, idx)                                    # p^n_i
        vi_s = bc_velocity(bc, (p_i + bar_dt * node_force(grid_read, idx)) / m,
                           i, j, k, lo, hi, padding)                             # v*_i
        v_star = v_star + N * vi_s
        B_star = B_star + B_update(spline, N, r_rel, vi_s)
        if uses_flip(scheme)
            vi_n    = p_i / m                                                   # v^n_i
            v_bar_n = v_bar_n + N * vi_n
            B_n     = B_n + B_update(spline, N, r_rel, vi_n)
        end
    end

    Minv   = M_inv(spline, inv_dx)
    C_star = B_star * Minv

    # ------------------------------ FLIP part ------------------------------- #
    if uses_flip(scheme)                              # APIC never reads v_p, C_p
        α     = flip_ratio(scheme, T)
        v_new = v_star + α * (particles.vel[p_idx] - v_bar_n)
        C_new = C_star + α * (particles.C[p_idx] - B_n * Minv)             # C-FLIP
    else
        v_new = v_star
        C_new = C_star
    end

    # v_p^{n+1/2}, C'_p (the spin-fix of this step is added in pass 2)
    particles.vel.x[p_idx] = v_new[1]
    particles.vel.y[p_idx] = v_new[2]
    particles.vel.z[p_idx] = v_new[3]
    particles.C[p_idx]     = C_new

    # wave speed of the kick, completed in pass 2
    particles.wave_speed[p_idx] = max(norm(v_new), norm(v_star)) +
                                  max(norm(C_star), norm(C_new)) * (1/inv_dx)

    # ------------------- P2G (MUSL) at the midpoint x^{n+1/2} ------------------- #
    pos_mid      = midpoint_position(particles, p_idx, dt)
    grid_pos_mid = grid_position(pos_mid, inv_dx, origin)
    position_ok(grid_write, grid_pos_mid) || return nothing     # counted in pass 2

    # C at t^{n+1/2}: half of the spin-fix impulse, predicted with ṽ^{n-1/2}
    C_mid    = C_new + spin_fix(Minv, dt / 2, particles.v_drift[p_idx], v_new)
    affine_p = mass * C_mid

    base_mid  = get_support_base(spline, grid_pos_mid)
    cache_mid = stencil_cache(grid_write, spline, base_mid)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_mid[1] + di
        j = base_mid[2] + dj
        k = base_mid[3] + dk

        idx, status = node_slot(grid_write, cache_mid, i, j, k)
        if status != NODE_OK
            status == NODE_INACTIVE && record_miss!(grid_write)
            continue
        end

        natural_coords = grid_pos_mid - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx)
        N = shapefunction(spline, natural_coords)

        scatter_musl!(musl, idx, N * mass, N * (mass * v_new + affine_p * r_rel))
    end

    return nothing
end


# ---------------------------------------------------------------------------- #
#     Pass 2: MUSL G2P at x^{n+1/2}, drift, F, material, P2G at x^{n+1}         #
# ---------------------------------------------------------------------------- #
@inline function g2p2g_pass2_particle!(
    p_idx, particles, material,
    grid_read, grid_write, musl,
    bc, lo, hi, padding,
    origin, inv_dx::T, spline, dt::T
) where {T}
    mass_cutoff = sqrt(floatmin(T))

    pos_old = particles.pos[p_idx]
    mass    = particles.mass[p_idx]
    V0      = particles.V0[p_idx]

    pos_mid      = midpoint_position(particles, p_idx, dt)     # same value as in pass 1
    grid_pos_mid = grid_position(pos_mid, inv_dx, origin)
    if !position_ok(grid_read, grid_position(pos_old, inv_dx, origin)) ||
       !position_ok(grid_write, grid_pos_mid)   # SparseGrid: NaN or outside the key range
        record_miss!(grid_write)
        return nothing                          # vel, C, wave_speed keep their pass-1 values
    end

    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    # ------------------- G2P from the MUSL field at x^{n+1/2} ------------------- #
    v_tilde = zero(SVector{3, T})
    B_tilde = zero(SMatrix{3, 3, T, 9})

    base_mid  = get_support_base(spline, grid_pos_mid)
    cache_mid = stencil_cache(grid_write, spline, base_mid)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_mid[1] + di
        j = base_mid[2] + dj
        k = base_mid[3] + dk

        idx, status = node_slot(grid_write, cache_mid, i, j, k)
        status == NODE_OK || continue
        m = musl_mass(musl, idx)
        m > mass_cutoff || continue

        natural_coords = grid_pos_mid - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx)
        N = shapefunction(spline, natural_coords)

        vi_t    = bc_velocity(bc, musl_momentum(musl, idx) / m, i, j, k, lo, hi, padding)   # ṽ_i
        v_tilde = v_tilde + N * vi_t
        B_tilde = B_tilde + B_update(spline, N, r_rel, vi_t)
    end

    Minv = M_inv(spline, inv_dx)
    L    = B_tilde * Minv                           # velocity gradient at x^{n+1/2}

    # ------------------------------- drift ---------------------------------- #
    # (an ASFLIP position correction β Δv would be added to v_adv here)
    v_adv   = v_tilde
    pos_new = pos_old + dt * v_adv
    particles.pos.x[p_idx] = pos_new[1]
    particles.pos.y[p_idx] = pos_new[2]
    particles.pos.z[p_idx] = pos_new[3]
    particles.v_drift.x[p_idx] = v_adv[1]          # predictor of the next midpoint
    particles.v_drift.y[p_idx] = v_adv[2]
    particles.v_drift.z[p_idx] = v_adv[3]

    # ------------------------------ material -------------------------------- #
    F_old     = particles.F[p_idx]
    mat_state = particles.mat_state[p_idx]

    ΔF    = exp_taylor2(dt * L)                     # midpoint exponential
    F_new = ΔF * F_old
    particles.F[p_idx] = F_new
    C_eff = (ΔF - one(ΔF)) / dt                     # = L + Δt/2 L²: materials use I + Δt C ≡ ΔF
    σ, mat_state_write = material_model(material, mat_state, F_new, C_eff, V0, mass, dt, 1/inv_dx)
    particles.mat_state[p_idx] = mat_state_write
    vol_new = det(F_new) * V0

    # ------------------------------ spin-fix -------------------------------- #
    # Advected with v_adv, carrying v_p: remove Δt m v_adv × v_p from the angular momentum.
    v_p   = SVector{3, T}(particles.vel.x[p_idx], particles.vel.y[p_idx], particles.vel.z[p_idx])
    C_p2g = particles.C[p_idx] + spin_fix(Minv, dt, v_adv, v_p)
    particles.C[p_idx] = C_p2g

    # wave speed for the CFL condition of the next step, stored per particle (no atomic max)
    particles.wave_speed[p_idx] = soundspeed(material, mat_state_write) +
                                  max(particles.wave_speed[p_idx],
                                      norm(v_tilde) + max(norm(L), norm(C_p2g)) * (1/inv_dx))

    affine_p = mass * C_p2g                  # momentum transfer
    affine_f = -(vol_new * σ * Minv)         # stress force, WITHOUT dt

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

        scatter!(grid_write, idx,
                 N * mass,
                 N * (mass * v_p + affine_p * r_rel),
                 N * (affine_f * r_rel))
    end

    return nothing
end


# ---------------------------------------------------------------------------- #
#                                 Kernels + host                               #
# ---------------------------------------------------------------------------- #
@kernel function g2p2g_pass1_kernel!(
    particles, scheme,
    grid_read, grid_write, musl,
    bc, lo, hi, padding,
    origin, inv_dx::T, spline, bar_dt::T, dt::T
) where {T}
    p_idx = @index(Global, Linear)
    g2p2g_pass1_particle!(p_idx, particles, scheme,
                          grid_read, grid_write, musl,
                          bc, lo, hi, padding,
                          origin, inv_dx, spline, bar_dt, dt)
end

@kernel function g2p2g_pass2_kernel!(
    particles, material,
    grid_read, grid_write, musl,
    bc, lo, hi, padding,
    origin, inv_dx::T, spline, dt::T
) where {T}
    p_idx = @index(Global, Linear)
    g2p2g_pass2_particle!(p_idx, particles, material,
                          grid_read, grid_write, musl,
                          bc, lo, hi, padding,
                          origin, inv_dx, spline, dt)
end


"""
    g2p2g!(model, dt, bar_dt = kick_dt(model.dt_prev, dt))

One staggered-leapfrog step on the particles: kick with `bar_dt`, drift with `dt`.
Reads `grid.state_read`, writes `grid.state_write` (external forces, walls and the
swap follow in `step!`). Does not change `model.t` or `model.dt_prev`.
"""
function g2p2g!(model::MPMModel{T}, dt::T, bar_dt::T = kick_dt(model.dt_prev, dt)) where {T}
    grid    = model.grid
    spline  = model.shapefunction
    scheme  = model.transfer_scheme
    bc      = model.boundary_condition
    backend = model.backend

    check_stencil(grid, spline)
    lo, hi = bc_bounds(bc, grid)     # errors for NoSlipBoundary on an OpenDomain

    prepare_musl!(grid)              # shaped like the write state, zeroed
    grid_read  = read_access(grid)
    grid_write = write_access(grid)
    musl       = musl_access(grid)

    # Pass 1: every particle must have deposited its MUSL mass/momentum before any
    # particle reads the MUSL field, hence two kernels (ordered on the backend).
    kernel_1 = g2p2g_pass1_kernel!(backend)
    foreach(model.particle_sets) do particle_set
        kernel_1(particle_set.particles, scheme,
                 grid_read, grid_write, musl,
                 bc, lo, hi, grid.padding,
                 grid.origin, grid.inv_dx, spline, bar_dt, dt;
                 ndrange=length(particle_set.particles))
    end
    KernelAbstractions.synchronize(backend)

    # Pass 2
    kernel_2 = g2p2g_pass2_kernel!(backend)
    foreach(model.particle_sets) do particle_set
        kernel_2(particle_set.particles, particle_set.material,
                 grid_read, grid_write, musl,
                 bc, lo, hi, grid.padding,
                 grid.origin, grid.inv_dx, spline, dt;
                 ndrange=length(particle_set.particles))
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end