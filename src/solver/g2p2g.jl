# ---------------------------------------------------------------------------- #
#                                     G2P2G                                    #
# ---------------------------------------------------------------------------- #
#
# 2-pass G2P2G: staggered leapfrog (kick-drift-kick, 2nd order in time) with a
# MUSL double mapping at the midpoint configuration and the AFLIP transfer
# (Fei et al. 2021) plus two additions that conserve angular momentum exactly:
#   * C-FLIP:   C'_p = C* + α (C_p - C̄^n)
#   * Spin-fix: the particle is advected with ṽ_p but carries v_p, which changes
#               the angular momentum by Δt m ṽ_p × v_p; this is booked into the
#               skew part of C.
# APIC is AFLIP with α ≡ 0 at compile time: every FLIP branch is removed by
# `uses_flip(scheme)`.
# Schemes with heat = true book the kinetic energy change of the transfer that is
# not work of the grid forces (and that of the spin-fix) into e (see transfer_schemes.jl).
#
# Time levels: v_p, C_p and the drift velocity ṽ_p live on half steps, x_p, F_p,
# σ_p and the grid on full steps. Δt_n = t^{n+1} - t^n,
# Δt̄_n = (Δt_{n-1} + Δt_n) / 2 (time-centered kick), Δt̄_0 = Δt_0 / 2.
#
#   read grid  (P2G at x^n):  m_I,  p_I = Σ N m (v^{n-1/2} + C r),  f_I(x^n, F^n) (+ m_I a_ext)
#
#   Pass 1  G2P at x^n from the read grid, P2G at x^{n+1/2} into the MUSL buffer
#     kick      v*_I = BC[(p_I + Δt̄_n f_I) / m_I],   v^n_I = p_I / m_I
#     AFLIP     v_p^{n+1/2} = v̄* + α (v_p^{n-1/2} - v̄^n),   C'_p = C* + α (C_p - C̄^n)
#     midpoint  x_p^{n+1/2} = x_p^n + Δt_n/2 ṽ_p^{n-1/2}                     (O(Δt²) predictor)
#     MUSL      m^musl_I, p^musl_I = Σ N(x^{n+1/2}) m (v_p^{n+1/2} + (C'_p + ½δC_p^pred) r)
#   Pass 2  G2P at x^{n+1/2} from the MUSL buffer, P2G at x^{n+1} into the write grid
#     ṽ_I = BC[p^musl_I / m^musl_I],   ṽ_p^{n+1/2},  L_p^{n+1/2} = ∇ṽ at x^{n+1/2}
#     drift     x_p^{n+1} = x_p^n + Δt_n ṽ_p^{n+1/2}                          (midpoint rule)
#     F         F_p^{n+1} = (I + Δt L + ½(Δt L)²) F_p^n ≈ exp(Δt L^{n+1/2}) F_p^n
#     material  σ_p^{n+1}, with C_eff = (ΔF - I)/Δt, so that I + Δt C_eff ≡ ΔF
#     spin-fix  C_p = C'_p + δC_p,   δC_p = ½[-M⁻¹ Δt_n (ṽ_p × v_p)]×
#     P2G       m, p = Σ N(x^{n+1}) m (v_p + C_p r),  f = -Σ V σ M⁻¹ r N  (WITHOUT Δt)
#   step!: external forces f_I += m_I a(x_I) (no Δt), walls, swap.
#
# Why each piece is needed for the 2nd order (checked in tests/test_time_order.jl):
#   * The MUSL velocity ṽ = Π_x(v) depends on the particle configuration x it is
#     mapped at (shape function weights). Evaluated at x^n instead of x^{n+1/2},
#     the drift has a local O(Δt² ∂Π/∂x) error: 1st order (with a small, O(Δx),
#     constant). The predictor only has to be O(Δt²) accurate; ṽ^{n-1/2} is.
#   * With the gradient L at the midpoint configuration, F needs the midpoint
#     exponential exp(Δt L) (Taylor-2 is enough); I + Δt L alone is 1st order.
#     (With the gradient at x^n it would be the other way round: I + Δt C̃
#     already is 2nd order and the ½(Δt C̃)² term makes it 1st order.)
#   * The spin-fix is an impulse at t^{n+1/2}, between two half-step values of C.
#     The MUSL map at t^{n+1/2} therefore uses the centered C'_p + ½δC_p, with
#     δC_p predicted from ṽ^{n-1/2}. The MUSL field only steers the drift; the
#     full, exact δC_p (with ṽ^{n+1/2}) goes into C_p and the P2G, so the angular
#     momentum stays exact. Without the centering the spin-fix is 1st order.
#   * External forces are kicked with Δt̄_n like the stress force (force field).
# For α < 1 (and APIC), the per-step PIC/APIC projection has no Δt -> 0 limit on a
# fixed grid, so the order in Δt is only meaningful for α = 1 or for a simultaneous
# refinement of Δt and Δx.
#
# The MUSL buffer has the layout of the write grid: its tiles were activated around
# x^n, so they contain the stencils at x^{n+1/2} and x^{n+1}.
#
# ONE kernel per pass for every grid type. The kernels only talk to the grid
# through a small "grid access" interface (src/grids/grid_access.jl). Julia
# compiles them separately for every access type, so the dense / sparse
# decisions are made at compile time; nothing is dispatched at run time.
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
    h_node  = zero(T)       # heat: share of the P2G losses of the nodes (specific)
    S2_star = zero(T)       # heat: Σ N |v*_i|² for the G2P residual

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
        if books_heat(scheme)
            # P2G loss of the node (Jensen, ≥ 0), shared by mass: N m_p / m_i of it
            L_i     = max(node_kin(grid_read, idx) - dot(p_i, p_i) / (2m), zero(T))
            h_node  = h_node + N * L_i / m
            S2_star = S2_star + N * dot(vi_s, vi_s)
        end
        if needs_bar_n(scheme)
            vi_n    = p_i / m                                                   # v^n_i
            v_bar_n = v_bar_n + N * vi_n
            B_n     = B_n + B_update(spline, N, r_rel, vi_n)
        end
    end

    Minv   = M_inv(spline, inv_dx)
    C_star = B_star * Minv

    # ------------------------------ FLIP part ------------------------------- #
    C_bar_n = B_n * Minv
    if uses_flip(scheme)                              # APIC never reads v_p, C_p
        α     = flip_ratio(scheme, T)
        v_new = v_star + α * (particles.vel[p_idx] - v_bar_n)
        C_new = C_star + α * (particles.C[p_idx] - C_bar_n)                # C-FLIP
    else
        v_new = v_star
        C_new = C_star
    end


    # ---- heat: kinetic energy the transfer dissipates -> e (all terms ≥ 0) -- #
    # Particle kinetic energy K = ½ m (|v|² + d‖C‖²) = Σ_i N ½ m |v + C r|². Per step
    #   APIC path:  K_old - (P2G loss of the nodes) + work - (G2P residual of the fit)
    #   AFLIP:      u_new = (1-α) ū* + α (u + ū* - ū)  (convex combination), so
    #               K_new = (1-α) K_APIC + α K_FLIP - ½ m α (1-α) ‖ū - u‖²
    # Booked: (1-α) (node loss share + G2P residual) + ½ α (1-α) ‖ū - u‖²  (specific).
    # Not booked (sign-indefinite): the energy error of the FLIP path (α term) and the
    # kinetic energy change of the spin-fix.
    if books_heat(scheme) && mass > zero(T)
        d   = one(T) / Minv                                   # D = d I (quadratic: Δx²/4)
        R   = max((S2_star - dot(v_star, v_star) - d * sum(abs2, C_star)) / 2, zero(T))
        α_h = flip_ratio(scheme, T)
        B   = uses_flip(scheme) ?
              α_h * (one(T) - α_h) / 2 * (sum(abs2, v_bar_n - particles.vel[p_idx]) +
                                          d * sum(abs2, C_bar_n - particles.C[p_idx])) : zero(T)
        Δe  = (one(T) - α_h) * (h_node + R) + B
        particles.mat_state[p_idx] = add_heat(particles.mat_state[p_idx], Δe)
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
    p_idx, particles, material, scheme,
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

    # spin-fix of this drift (applied to C below); its kinetic energy change -> heat
    v_p = SVector{3, T}(particles.vel.x[p_idx], particles.vel.y[p_idx], particles.vel.z[p_idx])
    C_o = particles.C[p_idx]
    δC  = spin_fix(Minv, dt, v_adv, v_p)

    ΔF    = exp_taylor2(dt * L)                     # midpoint exponential
    F_new = ΔF * F_old
    particles.F[p_idx] = F_new
    C_eff = (ΔF - one(ΔF)) / dt                     # = L + Δt/2 L²: materials use I + Δt C ≡ ΔF
    σ, mat_state_write = material_model(material, mat_state, F_new, C_eff, V0, mass, dt, 1/inv_dx)
    particles.mat_state[p_idx] = mat_state_write
    vol_new = det(F_new) * V0

    # ------------------------------ spin-fix -------------------------------- #
    # Advected with v_adv, carrying v_p: remove Δt m v_adv × v_p from the angular momentum.
    C_p2g = C_o + δC
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
        if books_heat(scheme)
            v_aff = v_p + C_p2g * r_rel
            scatter_kin!(grid_write, idx, N * mass * dot(v_aff, v_aff) / 2)
        end
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
    particles, material, scheme,
    grid_read, grid_write, musl,
    bc, lo, hi, padding,
    origin, inv_dx::T, spline, dt::T
) where {T}
    p_idx = @index(Global, Linear)
    g2p2g_pass2_particle!(p_idx, particles, material, scheme,
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
        kernel_2(particle_set.particles, particle_set.material, scheme,
                 grid_read, grid_write, musl,
                 bc, lo, hi, grid.padding,
                 grid.origin, grid.inv_dx, spline, dt;
                 ndrange=length(particle_set.particles))
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end
