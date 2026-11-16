# ---------------------------------------------------------------------------- #
#                                     G2P2G                                    #
# ---------------------------------------------------------------------------- #
#
# Fused G2P -> particle update -> P2G with the AFLIP transfer (Fei et al. 2021)
# plus two additions that conserve angular momentum exactly:
#   * C-FLIP:   C_p^{n+1} = C* + α (C_p^n - C̄^n)
#   * Spin-fix: the advection cross term dt (1-β) v̄* × Δv is booked into the
#               skew part of C (β = 0 for AFLIP, ASFLIP later).
# APIC is AFLIP with α ≡ 0 at compile time: every FLIP branch is removed by
# `uses_flip(scheme)`.
#
# Grid layout (all grids):
#   momentum = p^n   pure P2G momentum  Σ N m (v + C r)
#   force    = f     stress force + external forces, WITHOUT dt
# The G2P builds v*_i = BC[(p^n_i + dt f_i) / m_i] with the dt of the CURRENT step
# (clean leapfrog also for a variable dt) and v^n_i = p^n_i / m_i.
#
# ONE kernel for every grid type. The kernel only talks to the grid through a
# small "grid access" interface (src/grids/grid_access.jl). Julia compiles the kernel separately
# for every access type, so the dense / sparse decisions are made at compile
# time; nothing is dispatched at run time on the device.
#
# Replaces the old g2p2g.jl + g2p2g_sparse.jl.
# ---------------------------------------------------------------------------- #

# --------------------------- shared helpers --------------------------------- #
"""Skew matrix S = ½[a]× with axial vector ax(S) = a."""
@inline half_cross(a::SVector{3, T}) where {T} =
    SMatrix{3, 3, T, 9}(0, a[3] / 2, -a[2] / 2,
                        -a[3] / 2, 0, a[1] / 2,
                        a[2] / 2, -a[1] / 2, 0)

"""Boundary condition applied to the node velocity v*_i in the G2P."""
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
#                         Device function (one particle)                       #
# ---------------------------------------------------------------------------- #
# Early returns are fine here; the kernel below is only a thin wrapper.
@inline function g2p2g_particle!(
    p_idx, particles, material, scheme,
    grid_read, grid_write,
    bc, lo, hi, padding,
    origin, inv_dx::T, spline, dt::T
) where {T}
    mass_cutoff = sqrt(floatmin(T))  # Threshold to avoid division by zero in G2P

    # Extract particle properties
    pos_old = particles.pos[p_idx]
    mass    = particles.mass[p_idx]
    V0      = particles.V0[p_idx]

    grid_pos_old = grid_position(pos_old, inv_dx, origin)
    if !position_ok(grid_read, grid_pos_old)    # SparseGrid: NaN or outside the key range
        record_miss!(grid_write)
        return nothing                          # vel, C, wave_speed keep their old values
    end

    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    v_star  = zero(SVector{3, T});       v_bar_n = zero(SVector{3, T})
    B_star  = zero(SMatrix{3, 3, T, 9}); B_n     = zero(SMatrix{3, 3, T, 9})

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
        m > mass_cutoff || continue

        natural_coords = grid_pos_old - SVector(i, j, k)
        r_rel = - natural_coords * (1 / inv_dx)
        N = shapefunction(spline, natural_coords)

        p_i  = node_momentum(grid_read, idx)                                  # p^n_i
        vi_s = bc_velocity(bc, (p_i + dt * node_force(grid_read, idx)) / m,
                           i, j, k, lo, hi, padding)                           # v*_i
        v_star = v_star + N * vi_s
        B_star = B_star + B_update(spline, N, r_rel, vi_s)
        if uses_flip(scheme)
            vi_n    = p_i / m                                                 # v^n_i
            v_bar_n = v_bar_n + N * vi_n
            B_n     = B_n + B_update(spline, N, r_rel, vi_n)
        end
    end

    Minv   = M_inv(spline, inv_dx)
    C_star = B_star * Minv

    # ------------------------------ FLIP part ------------------------------- #
    β = advection_ratio(scheme, T)                    # AFLIP: 0
    if uses_flip(scheme)                              # APIC never reads v_p, C_p
        α     = flip_ratio(scheme, T)
        Δv    = α * (particles.vel[p_idx] - v_bar_n)
        C_p2g = C_star + α * (particles.C[p_idx] - B_n * Minv) +             # C-FLIP
                half_cross(-Minv * dt * (one(T) - β) * cross(v_star, Δv))  # spin-fix
    else
        Δv    = zero(SVector{3, T})
        C_p2g = C_star
    end
    v_new = v_star + Δv

    # all schemes store v_p and C_p
    particles.vel.x[p_idx] = v_new[1]
    particles.vel.y[p_idx] = v_new[2]
    particles.vel.z[p_idx] = v_new[3]
    particles.C[p_idx]     = C_p2g

    # ------------------------------- advection ------------------------------ #
    pos_new = pos_old + dt * (v_star + β * Δv)
    particles.pos.x[p_idx] = pos_new[1]
    particles.pos.y[p_idx] = pos_new[2]
    particles.pos.z[p_idx] = pos_new[3]

    # ------------------------- material (with C*) --------------------------- #
    F_old     = particles.F[p_idx]
    mat_state = particles.mat_state[p_idx]

    F_new = (one(SMatrix{3, 3, T, 9}) + C_star * dt) * F_old
    particles.F[p_idx] = F_new
    σ, mat_state_write = material_model(material, mat_state, F_new, C_star, V0, mass, dt, 1/inv_dx)
    particles.mat_state[p_idx] = mat_state_write
    vol_new = det(F_new) * V0

    # wave speed for the CFL condition of the next step, stored per particle (no atomic max)

    particles.wave_speed[p_idx] = soundspeed(material, mat_state_write) +
                                  max(norm(v_new), norm(v_star)) +
                                  max(norm(C_star), norm(C_p2g)) * (1/inv_dx)

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
                 N * (mass * v_new + affine_p * r_rel),
                 N * (affine_f * r_rel))
    end

    return nothing
end


# ---------------------------------------------------------------------------- #
#                                 Kernel + host                                #
# ---------------------------------------------------------------------------- #
@kernel function g2p2g_kernel!(
    particles, material, scheme,
    grid_read, grid_write,
    bc, lo, hi, padding,
    origin, inv_dx::T, spline, dt::T
) where {T}
    p_idx = @index(Global, Linear)
    g2p2g_particle!(p_idx, particles, material, scheme,
                    grid_read, grid_write,
                    bc, lo, hi, padding,
                    origin, inv_dx, spline, dt)
end


function g2p2g!(model::MPMModel{T}, dt::T) where {T}
    grid    = model.grid
    spline  = model.shapefunction
    scheme  = model.transfer_scheme
    bc      = model.boundary_condition
    backend = model.backend

    check_stencil(grid, spline)
    lo, hi = bc_bounds(bc, grid)     # errors for NoSlipBoundary on an OpenDomain

    grid_read  = read_access(grid)
    grid_write = write_access(grid)
    kernel     = g2p2g_kernel!(backend)

    foreach(model.particle_sets) do particle_set
        kernel(particle_set.particles, particle_set.material, scheme,
               grid_read, grid_write,
               bc, lo, hi, grid.padding,
               grid.origin, grid.inv_dx, spline, dt;
               ndrange=length(particle_set.particles))
    end

    KernelAbstractions.synchronize(backend)
    return nothing
end