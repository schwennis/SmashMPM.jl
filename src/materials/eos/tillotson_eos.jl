# ---------------------------------------------------------------------------- #
#                          Tillotson Equation of State                         #
# ---------------------------------------------------------------------------- #
@kwdef struct TillotsonEOS{T} <: AbstractEquationOfState
    ρ::T   # Reference density
    A::T    # Tillotson parameter A
    B::T    # Tillotson parameter B
    α::T    # Tillotson parameter α
    β::T    # Tillotson parameter β
    E0::T   # Reference energy
    Eiv::T  # Initial energy for condensed state
    Ecv::T  # Critical energy for expanded state
    a::T    # Tillotson parameter a
    b::T    # Tillotson parameter b
end

@kwdef struct TillotsonState{T} <: AbstractEoSState
    p::T    # Pressure
end


# Initialization and reference sound speed
init_state(::TillotsonEOS{T}) where {T} = TillotsonState(zero(T))
reference_soundspeed(eos::TillotsonEOS) = sqrt(eos.A / eos.ρ)


# ---------------------------------------------------------------------------- #
#                           update_eos implementation                          #
# ---------------------------------------------------------------------------- #
@inline function update_eos(eos::TillotsonEOS{T}, ::TillotsonState{T}, ρ::T, e::T) where {T}
    p, c = tillotson_eos_and_soundspeed(eos, ρ, e)
    return TillotsonState(p), c
end

@inline function tillotson_eos_and_soundspeed(eos::TillotsonEOS{T}, ρ::T, e::T) where {T}
    ρ = eos.ρ
    Eiv = eos.Eiv
    Ecv = eos.Ecv

    ρ_safe = max(T(ρ), eps(T))
    e_safe = max(T(e), zero(T))

    if ρ_safe >= ρ
        p, dp_dρ, dp_de = _tillotson_condensed(eos, ρ_safe, e_safe)
    elseif e_safe < Eiv
        p, dp_dρ, dp_de = _tillotson_condensed(eos, ρ_safe, e_safe)
    elseif e_safe > Ecv
        p, dp_dρ, dp_de = _tillotson_expanded(eos, ρ_safe, e_safe)
    else
        pc, dp_dρ_c, dp_de_c = _tillotson_condensed(eos, ρ_safe, e_safe)
        pe, dp_dρ_e, dp_de_e = _tillotson_expanded(eos, ρ_safe, e_safe)

        ΔE  = Ecv - Eiv
        w_e = (e_safe - Eiv) / ΔE
        w_c = one(T) - w_e

        p     = w_e * pe + w_c * pc
        dp_dρ = w_e * dp_dρ_e + w_c * dp_dρ_c
        dp_de = w_e * dp_de_e + w_c * dp_de_c + (pe - pc) / ΔE
    end

    c_squared = max(dp_dρ + (p / ρ_safe^2) * dp_de, zero(T))
    return p, sqrt(c_squared)
end

# ---------------------------------------------------------------------------- #
#                                    Helpers                                   #
# ---------------------------------------------------------------------------- #
@inline function _tillotson_omega(eos::TillotsonEOS{T}, ρ::T, e::T) where {T}
    b  = eos.b
    E0 = eos.E0
    ρ = eos.ρ

    η = ρ / ρ

    denom = one(T) + e / (E0 * η^2)
    ω = b / denom

    ddenom_dρ = -T(2) * e / (E0 * η^3 * ρ)
    dω_dρ = -b * ddenom_dρ / denom^2

    ddenom_de = one(T) / (E0 * η^2)
    dω_de = -b * ddenom_de / denom^2

    return ω, dω_dρ, dω_de
end

@inline function _tillotson_condensed(eos::TillotsonEOS{T}, ρ::T, e::T) where {T}
    ρ = eos.ρ
    A  = eos.A
    B  = eos.B
    a  = eos.a

    η = ρ / ρ
    μ = η - one(T)

    ω, dω_dρ, dω_de = _tillotson_omega(eos, ρ, e)

    p = (a + ω) * ρ * e + A * μ + B * μ^2

    dp_dρ = (a + ω) * e + ρ * e * dω_dρ + (A + T(2) * B * μ) / ρ
    dp_de = (a + ω) * ρ + ρ * e * dω_de

    return p, dp_dρ, dp_de
end

@inline function _tillotson_expanded(eos::TillotsonEOS{T}, ρ::T, e::T) where {T}
    ρ = eos.ρ
    A  = eos.A
    α  = eos.α
    β  = eos.β
    a  = eos.a

    η = ρ / ρ
    μ = η - one(T)
    ν = ρ / ρ - one(T)

    ω, dω_dρ, dω_de = _tillotson_omega(eos, ρ, e)

    dν_dρ = -ρ / ρ^2
    dμ_dρ = one(T) / ρ

    exp_term_1 = exp(-β * ν)
    exp_term_2 = exp(-α * ν^2)

    P_star = ω * ρ * e + A * μ * exp_term_1
    p = a * ρ * e + P_star * exp_term_2

    dP_star_dρ = e * (ω + ρ * dω_dρ) + A * exp_term_1 * (dμ_dρ - β * μ * dν_dρ)
    dP_star_de = ρ * (ω + e * dω_de)

    dp_dρ = a * e + (dP_star_dρ - T(2) * α * ν * dν_dρ * P_star) * exp_term_2
    dp_de = a * ρ + dP_star_de * exp_term_2

    return p, dp_dρ, dp_de
end