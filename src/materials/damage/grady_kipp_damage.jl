# ---------------------------------------------------------------------------- #
#                            Grady-Kipp Damage Model                           #
# ---------------------------------------------------------------------------- #
# Explicit Weibull flaws after Benz & Asphaug, Icarus 107 (1994) 98 and
# Comput. Phys. Commun. 87 (1995) 253.
#
# Flaw distribution (once per body, on the host):
#   The flaws j = 1, 2, 3, ... get the activation strains of the Weibull
#   distribution n(ε) = k ε^m of the whole body (volume V),
#       ε_j = (j / (k V))^(1/m),
#   and are each assigned to a randomly chosen particle, until every particle
#   has at least one flaw (on average N_flaw ≈ n_p ln n_p flaws in total).
#   The weakest flaw of the body is (1 / (k V))^(1/m), independent of n_p.
#
# Evolution (per particle):
#   ε       = σ_max / ((1 - D) E)                   local tensile strain
#   n_act   = #{ flaws of the particle with ε_j < ε }
#   dD^⅓/dt = n_act c_g / R_s,   c_g = 0.4 c_l      (c_l: longitudinal wave speed)
#   D^⅓    <= (n_act / n_f)^⅓                       a particle is only completely
#                                                   destroyed once all its flaws are active
#
# Storage: the number of flaws differs from particle to particle. They are kept in
# ONE flat array of the damage model (CSR layout); the particle state only holds
# the index of its first flaw and the number of its flaws. The index travels with
# the particle when the particle arrays are reordered. Since the flaws are assigned
# in the order of increasing j, the thresholds of every particle are sorted.
# The thresholds are stored in Float32 (half the memory, the relative error of
# 6e-8 is irrelevant for a statistical threshold).
#
# The Weibull parameters are only needed for the sampling on the host and are
# kept in Float64, since k easily exceeds floatmax(Float32).
struct GradyKippDamage{T, V<:AbstractVector{Float32}} <: AbstractDamage
    E::T                    # Young's modulus
    m::Float64              # Weibull exponent
    k::Float64              # Weibull coefficient (1/m^3)
    cg_factor::T            # crack growth speed c_g = cg_factor * c_l
    D_frac::T               # D >= D_frac: particle counts as fractured (isotropic F, no q in expansion)
    J_max::T                # upper bound of det F for fractured particles
    flaws::V                # activation strains of all particles (CSR), filled by init_states
end

"""
    GradyKippDamage{T}(; E, m, k, cg_factor = 0.4, D_frac = 0.99, J_max = 10)

Grady-Kipp damage with explicit Weibull flaws (Benz & Asphaug 1994, 1995).
The flaws are distributed when the body is built (`init_states`), the instance
created here has none yet.
"""
function GradyKippDamage{T}(; E, m, k, cg_factor = 0.4, D_frac = 0.99, J_max = 10.0,
                            flaws::AbstractVector{Float32} = Float32[]) where {T}
    return GradyKippDamage(T(E), Float64(m), Float64(k), T(cg_factor), T(D_frac), T(J_max), flaws)
end

# The flaw array has to move to the device (CuArray -> CuDeviceArray in the kernel)
Adapt.@adapt_structure GradyKippDamage

struct GradyKippState{T}
    D::T                # damage
    first_flaw::Int32   # index of the first flaw of the particle in dm.flaws
    n_flaws::Int32      # number of flaws of the particle (0: flawless, never damaged)
end


# Flawless state, used for type inference and if no flaws are distributed
init_state(::GradyKippDamage{T}) where {T} = GradyKippState(zero(T), one(Int32), zero(Int32))

# A single particle cannot be initialized on its own: the flaws belong to the body
init_state(::GradyKippDamage, rng, n_p, V) =
    error("GradyKippDamage: the flaws are distributed over the whole body, " *
          "use init_states / initial_material_states instead of a per-particle initialization.")

"""
    init_states(dm::GradyKippDamage, rng, n_p, V) -> (dm_with_flaws, states)

Distributes the flaws over the `n_p` particles of a body with volume `V`
(Benz & Asphaug 1995) and returns the damage model holding the flaws together
with the states of all particles.
"""
function init_states(dm::GradyKippDamage{T}, rng, n_p::Integer, V::Real) where {T}
    flaws, first_flaw, n_flaws = distribute_flaws(rng, n_p, dm.k * Float64(V), dm.m)
    states = [GradyKippState(zero(T), first_flaw[i], n_flaws[i]) for i in 1:n_p]
    dm_new = GradyKippDamage(dm.E, dm.m, dm.k, dm.cg_factor, dm.D_frac, dm.J_max, flaws)
    return dm_new, states
end

"""
    distribute_flaws(rng, n_p, kV, m) -> (flaws, first_flaw, n_flaws)

Benz & Asphaug (1995): flaw j = 1, 2, ... gets the activation strain
ε_j = (j / kV)^(1/m) and is assigned to a random particle, until every particle has
at least one flaw. CSR result: the flaws of particle i are
`flaws[first_flaw[i] : first_flaw[i] + n_flaws[i] - 1]`, sorted ascending.

Two passes over the same random numbers (copy of the rng): the first one counts
the flaws per particle, the second one writes them. Thus no N_flaw sized list of
owners is needed.
"""
function distribute_flaws(rng, n_p::Integer, kV::Float64, m::Float64)
    rng_replay = copy(rng)

    # 1. pass: number of flaws per particle, until no particle is left without one
    n_flaws = zeros(Int32, n_p)
    n_empty = n_p
    n_total = 0
    while n_empty > 0
        i = rand(rng, 1:n_p)
        n_empty -= iszero(n_flaws[i])
        n_flaws[i] += one(Int32)
        n_total += 1
    end
    n_total < typemax(Int32) || error("GradyKippDamage: too many flaws ($n_total) for Int32 indices, split the body.")

    # offsets of the particles in the flat array
    first_flaw = Vector{Int32}(undef, n_p)
    offset = one(Int32)
    for i in 1:n_p
        first_flaw[i] = offset
        offset += n_flaws[i]
    end

    # 2. pass: same draws again, flaw j goes to the next free slot of its particle
    flaws = Vector{Float32}(undef, n_total)
    slot  = copy(first_flaw)
    inv_m = 1 / m
    for j in 1:n_total
        i = rand(rng_replay, 1:n_p)
        flaws[slot[i]] = Float32((j / kV)^inv_m)
        slot[i] += one(Int32)
    end
    return flaws, first_flaw, n_flaws
end


damage_variable(st::GradyKippState) = st.D

@inline degradation(::GradyKippDamage, st::GradyKippState) = one(st.D) - st.D
@inline degrade_pressure(::GradyKippDamage, st::GradyKippState, p) = p < zero(p) ? (one(st.D) - st.D) * p : p
@inline is_fractured(dm::GradyKippDamage, st::GradyKippState) = st.D >= dm.D_frac
@inline fractured_max_J(dm::GradyKippDamage) = dm.J_max


@inline function update_damage(dm::GradyKippDamage, st::GradyKippState{T}, s, _pl, th,
                               kin::Kinematics{T}, c) where {T}
    n_f = st.n_flaws
    n_f > zero(Int32) || return st                      # flawless particle

    # local tensile strain of the (already damaged) particle
    g     = degradation(dm, st)
    σ     = g * s - degrade_pressure(dm, st, T(th.p)) * I
    ε_eff = max_principal_stress(σ) / (max(g, eps(T)) * dm.E)

    # number of active flaws: the thresholds are sorted, so counting stops at the
    # first one that is not reached
    flaws = dm.flaws
    i0    = st.first_flaw
    @inbounds ε_eff > flaws[i0] || return st
    n_act = one(Int32)
    @inbounds while n_act < n_f && ε_eff > flaws[i0 + n_act]
        n_act += one(Int32)
    end

    # growth of the crack volume, R_s: radius of the current particle volume
    R_s  = cbrt(3 * det(kin.F) * kin.V0 / (4 * T(π)))
    ΔD13 = kin.dt * n_act * dm.cg_factor * c / R_s

    # damage never heals, but may only grow up to the fraction of active flaws
    D13 = cbrt(st.D)
    D13 = max(D13, min(D13 + ΔD13, cbrt(T(n_act) / T(n_f))))
    return GradyKippState(D13^3, st.first_flaw, st.n_flaws)
end


# Largest eigenvalue of a symmetric 3x3 matrix (analytic, GPU-safe)
@inline function max_principal_stress(σ::SMatrix{3,3,T,9}) where {T}
    q  = tr(σ) / 3
    p1 = σ[1,2]^2 + σ[1,3]^2 + σ[2,3]^2
    p2 = (σ[1,1] - q)^2 + (σ[2,2] - q)^2 + (σ[3,3] - q)^2 + 2p1
    p  = sqrt(p2 / 6)
    iszero(p) && return q

    r = clamp(det((σ - q * I) / p) / 2, -one(T), one(T))
    return q + 2p * cos(acos(r) / 3)
end