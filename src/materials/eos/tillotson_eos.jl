# ---------------------------------------------------------------------------- #
#                          Tillotson Equation of State                         #
# ---------------------------------------------------------------------------- #

# Struct Definition
struct TillotsonEOS{T} <: AbstractEquationOfState
    ρ0::T   # Reference density
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

struct TillotsonState{T} <: AbstractEoSState
    p::T    # Pressure
    e::T    # Specific internal energy
end

function init_eos_state(eos::TillotsonEOS{T}) where {T}
    # Start with p = 0, e =0
    p0 = zero(T)
    e0 = zero(T)
    state = TillotsonState{T}(p0, e0)
    
    # Reference for Tillotson: c0 = sqrt(A / ρ0)
    c0 = sqrt(eos.A / eos.ρ0)
    
    return state, c0
end



# ---------------------------------------------------------------------------- #
#                           update_eos implementation                          #
# ---------------------------------------------------------------------------- #
function update_eos(eos::TillotsonEOS{T}, eos_state::TillotsonState{T}, ρ::T, stress_work::T, dt::T) where {T}
    e_new = eos_state.e + stress_work * dt
    p, c = tillotson_eos_and_soundspeed(eos, ρ, e_new)
    return TillotsonState{T}(p, e_new), c
end



function tillotson_eos_and_soundspeed(eos::TillotsonEOS{T}, ρ::T, e::T) where {T}
    ρ0 = eos.ρ0
    Eiv = eos.Eiv
    Ecv = eos.Ecv

    ρ_safe = max(T(ρ), eps(T))
    e_safe = max(T(e), zero(T))

    if ρ_safe >= ρ0
        p, dp_dρ, dp_de = _tillotson_condensed(eos, ρ_safe, e_safe)
    elseif e_safe < Eiv
        p, dp_dρ, dp_de = _tillotson_condensed(eos, ρ_safe, e_safe)
    elseif e_safe > Ecv
        p, dp_dρ, dp_de = _tillotson_expanded(eos, ρ_safe, e_safe)
    else
        pc, dp_dρ_c, dp_de_c = _tillotson_condensed(eos, ρ_safe, e_safe)
        pe, dp_dρ_e, dp_de_e = _tillotson_expanded(eos, ρ_safe, e_safe)

        ΔE  = eos.Ecv - eos.Eiv
        w_e = (e_safe - eos.Eiv) / ΔE
        w_c = one(T) - w_e

        p     = w_e * pe + w_c * pc
        dp_dρ = w_e * dp_dρ_e + w_c * dp_dρ_c
        dp_de = w_e * dp_de_e + w_c * dp_de_c + (pe - pc) / ΔE
    end

    c_squared = dp_dρ + (p / ρ_safe^2) * dp_de
    c_squared = max(c_squared, zero(T))
    c = sqrt(c_squared)

    return p, c
end



# ---------------------------------------------------------------------------- #
#                                    Helpers                                   #
# ---------------------------------------------------------------------------- #
"""
Compute the Tillotson omega function and its derivatives with respect to density and energy.
"""
function _tillotson_omega(eos::TillotsonEOS{T}, ρ::T, e::T) where {T}
    b = eos.b
    E0 = eos.E0
    ρ0 = eos.ρ0

    η = ρ / ρ0

    denom = one(T) + e / (E0 * η^2)
    ω = b / denom

    ddenom_dρ = -T(2) * e / (E0 * η^3 * ρ0) 
    dω_dρ = -b * ddenom_dρ / denom^2

    ddenom_de = one(T) / (E0 * η^2)
    dω_de = -b * ddenom_de / denom^2

    return ω, dω_dρ, dω_de
end

"""
Compute the pressure and its derivatives for the Tillotson equation of state in the condensed regime.
"""
function _tillotson_condensed(eos::TillotsonEOS{T}, ρ::T, e::T) where {T}
    ρ0 = eos.ρ0
    A = eos.A
    B = eos.B
    a = eos.a

    η = ρ / ρ0
    μ = η - one(T)

    ω, dω_dρ, dω_de = _tillotson_omega(eos, ρ, e)

    p = (a + ω) * ρ * e + A * μ + B * μ^2

    dp_dρ = (a + ω) * e + ρ * e * dω_dρ + (A + T(2) * B * μ) / ρ0
    dp_de = (a + ω) * ρ + ρ * e * dω_de

    return p, dp_dρ, dp_de
end


"""
Compute the pressure and its derivatives for the Tillotson equation of state in the expanded regime.
"""
function _tillotson_expanded(eos::TillotsonEOS{T}, ρ::T, e::T) where {T}
    ρ0 = eos.ρ0
    A = eos.A
    α = eos.α
    β = eos.β
    a = eos.a

    η = ρ / ρ0
    μ = η - one(T)
    ν = ρ0 / ρ - one(T)

    ω, dω_dρ, dω_de = _tillotson_omega(eos, ρ, e)

    dν_dρ = -ρ0 / ρ^2
    dμ_dρ = one(T) / ρ0

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