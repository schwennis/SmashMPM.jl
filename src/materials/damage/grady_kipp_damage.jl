# ---------------------------------------------------------------------------- #
#                            Grady-Kipp Damage Model                           #
# ---------------------------------------------------------------------------- #
# Weibull flaw distribution with resolution-independent initialization,
# as described in Yan et al., Icarus 455 (2026) 117080, Section 3.3
#
#   ε_eff  = σ_max / ((1 - D) E)
#   n_act  = (ε_eff / ε_min)^m_p                 (capped at n_f)
#   dD^⅓/dt = n_act c_g / R_s,   c_g = 0.4 c_p   (capped at (n_act / n_f)^⅓)
#
# Weibull parameters are only needed for the sampling on the host and are kept
# in Float64, since k easily exceeds floatmax(Float32).
@kwdef struct GradyKippDamage{T} <: AbstractDamage
    E::T                    # Young's modulus
    m::Float64              # Weibull exponent
    k::Float64              # Weibull coefficient (1/m^3)
    n_tot::Float64          # total number of flaws in the body
    cg_factor::T = 0.4      # crack growth speed c_g = cg_factor * c_p
end

struct GradyKippState{T}
    D::T            # damage
    ε_min::T        # smallest activation strain
    m_p::T          # Weibull exponent of the particle
    n_f::T          # number of flaws in the particle
end


# Flawless state, used for type inference and if no sampling is done
init_state(::GradyKippDamage{T}) where {T} = GradyKippState(zero(T), T(Inf), zero(T), one(T))

"""
    init_state(dm::GradyKippDamage, rng, n_p, V) -> GradyKippState

Samples the flaws of one of the `n_p` particles of a body with volume `V`.
"""
function init_state(dm::GradyKippDamage{T}, rng, n_p::Integer, V::Real) where {T}
    m, n_tot = dm.m, dm.n_tot
    kV  = dm.k * Float64(V)
    P_s = 1 / n_p
    activation_strain(N) = (N / kV)^(1 / m)

    n_tot > 10^m      || @warn "GradyKippDamage: n_tot should exceed 10^m" maxlog=1
    n_tot > 100 * n_p || @warn "GradyKippDamage: n_tot should exceed 100 n_p" maxlog=1

    N_min = 1 + rand(rng) * (10^m - 1)
    N_max = max(n_tot - log(1 - rand(rng)) / log1p(-P_s), nextfloat(N_min))
    n_f   = max(round(P_s * n_tot + sqrt(P_s * n_tot * (1 - P_s)) * randn(rng)), 1.0)

    ε_min = activation_strain(N_min)
    ε_max = activation_strain(N_max)
    m_p   = log(n_f) / log(ε_max / ε_min)

    return GradyKippState(zero(T), T(ε_min), T(m_p), T(n_f))
end

damage_variable(st::GradyKippState) = st.D

@inline degradation(::GradyKippDamage, st::GradyKippState) = one(st.D) - st.D
@inline degrade_pressure(::GradyKippDamage, st::GradyKippState, p) = p < zero(p) ? (one(st.D) - st.D) * p : p


@inline function update_damage(dm::GradyKippDamage, st::GradyKippState{T}, s, _pl, th,
                               kin::Kinematics{T}, c) where {T}
    g     = degradation(dm, st)
    σ     = g * s - degrade_pressure(dm, st, T(th.p)) * I
    ε_eff = max_principal_stress(σ) / (max(g, eps(T)) * dm.E)
    ε_eff > st.ε_min || return st

    n_act = min((ε_eff / st.ε_min)^st.m_p, st.n_f)
    R_s   = cbrt(3 * det(kin.F) * kin.V0 / (4 * T(π)))
    ΔD13  = kin.dt * n_act * dm.cg_factor * c / R_s

    D13 = cbrt(st.D)
    D13 = max(D13, min(D13 + ΔD13, cbrt(n_act / st.n_f)))
    return GradyKippState(D13^3, st.ε_min, st.m_p, st.n_f)
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