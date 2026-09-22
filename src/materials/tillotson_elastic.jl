# ---------------------------------------------------------------------------- #
#           Elastic material model using Tillotson equation of state           #
# ---------------------------------------------------------------------------- #

# Material parameters
struct TillotsonElastic{T} <: AbstractMaterial
    μ::T    # Shear modulus
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

# Corresponding material state
struct TillotsonElasticState{T} <: AbstractMaterialState
    p::T    # Pressure
    e::T    # Specific internal energy
    c::T    # Soundspeed
    s::SMatrix{3,3,T,9} # Deviatoric stress
end

# soundspeed helper for courant condition
function get_soundspeed(material::TillotsonElastic{T}, mat_state::TillotsonElasticState{T}) where {T}
    return mat_state.c
end


# ---------------------------------------------------------------------------- #
#                                Material model                                #
# ---------------------------------------------------------------------------- #
function material_model(material::TillotsonElastic{T}, mat_state::TillotsonElasticState{T}, F, C, V0, m, dt) where {T}
    # Kinematic part
    J = det(F)
    ρ = m / (J * V0)

    D = 0.5 * (C + C')
    W = 0.5 * (C - C')
    trD = tr(D)
    D_dev = D - (trD / 3) * one(SMatrix{3,3,T,9})

    s_old = mat_state.s
    s_rot = W * s_old - s_old * W
    s_new = s_old + dt * (2 * material.μ * D_dev + s_rot)

    p_current = mat_state.p
    stress_work = (-p_current * trD + dot(s_new, D_dev)) / ρ
    e_new = mat_state.e + stress_work * dt

    # EOS part
    p, c = tillotson_eos_and_soundspeed(material, ρ, e_new)

    # Assemble the Cauchy stress tensor
    σ = s_new - p * one(SMatrix{3,3,T,9})
    
    # Construct new material state
    mat_state = TillotsonElasticState{T}(p, e_new, c, s_new)

    return σ, mat_state
end

"""
    tillotson_eos_and_soundspeed(material, ρ, e)

    Compute the pressure and soundspeed for the Tillotson equation of state.

    # Arguments
    - `material`: The TillotsonElastic material
    - `ρ`: The density
    - `e`: The specific internal energy

    # Returns
    - `p`: The pressure
    - `c`: The soundspeed
"""
function tillotson_eos_and_soundspeed(material::TillotsonElastic{T}, ρ, e) where {T}
    ρ0 = material.ρ0
    Eiv = material.Eiv
    Ecv = material.Ecv

    ρ_safe = max(T(ρ), eps(T))
    e_safe = max(T(e), zero(T))

    if ρ_safe >= ρ0
        p, dp_dρ, dp_de = _tillotson_condensed(material, ρ_safe, e_safe)
    elseif e_safe < Eiv
        p, dp_dρ, dp_de = _tillotson_condensed(material, ρ_safe, e_safe)
    elseif e_safe > Ecv
        p, dp_dρ, dp_de = _tillotson_expanded(material, ρ_safe, e_safe)
    else
        pc, dp_dρ_c, dp_de_c = _tillotson_condensed(material, ρ_safe, e_safe)
        pe, dp_dρ_e, dp_de_e = _tillotson_expanded(material, ρ_safe, e_safe)

        ΔE  = material.Ecv - material.Eiv
        w_e = (e_safe - material.Eiv) / ΔE
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
function _tillotson_omega(material::TillotsonElastic{T}, ρ, e) where {T}
    b = material.b
    E0 = material.E0
    ρ0 = material.ρ0

    η = ρ / ρ0

    denom = one(T) + e / (E0 * η^2)
    ω = b / denom

    ddenom_dρ = -2 * e / (E0 * η^3 * ρ0) 
    dω_dρ = -b * ddenom_dρ / denom^2

    ddenom_de = one(T) / (E0 * η^2)
    dω_de = -b * ddenom_de / denom^2

    return ω, dω_dρ, dω_de
end

"""
Compute the pressure and its derivatives for the Tillotson equation of state in the condensed regime.
"""
function _tillotson_condensed(material::TillotsonElastic{T}, ρ, e) where {T}
    ρ0 = material.ρ0
    A = material.A
    B = material.B
    a = material.a

    η = ρ / ρ0
    μ = η - one(T)

    ω, dω_dρ, dω_de = _tillotson_omega(material, ρ, e)

    p = (a + ω) * ρ * e + A * μ + B * μ^2

    dp_dρ = (a + ω) * e + ρ * e * dω_dρ + (A + 2 * B * μ) / ρ0
    dp_de = (a + ω) * ρ + ρ * e * dω_de

    return p, dp_dρ, dp_de
end


"""
Compute the pressure and its derivatives for the Tillotson equation of state in the expanded regime.
"""
function _tillotson_expanded(material::TillotsonElastic{T}, ρ, e) where {T}
    ρ0 = material.ρ0
    A = material.A
    α = material.α
    β = material.β
    a = material.a

    η = ρ / ρ0
    μ = η - one(T)
    ν = ρ0 / ρ - one(T)

    ω, dω_dρ, dω_de = _tillotson_omega(material, ρ, e)

    dν_dρ = -ρ0 / ρ^2
    dμ_dρ = one(T) / ρ0

    exp_term_1 = exp(-β * ν)
    exp_term_2 = exp(-α * ν^2)

    P_star = ω * ρ * e + A * μ * exp_term_1

    p = a * ρ * e + P_star * exp_term_2

    dP_star_dρ = e * (ω + ρ * dω_dρ) + A * exp_term_1 * (dμ_dρ - β * μ * dν_dρ)
    dP_star_de = ρ * (ω + e * dω_de)

    dp_dρ = a * e + (dP_star_dρ - 2 * α * ν * dν_dρ * P_star) * exp_term_2
    dp_de = a * ρ + dP_star_de * exp_term_2

    return p, dp_dρ, dp_de
end