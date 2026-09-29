# ---------------------------------------------------------------------------- #
#                                Abstract Types                                #
# ---------------------------------------------------------------------------- #
abstract type AbstractMaterial end
abstract type AbstractMaterialState end

# Equation of State
abstract type AbstractEquationOfState end
abstract type AbstractEoSState end
init_eos_state(eos::AbstractEquationOfState) = error("init_eos_state not implemented for $(typeof(eos))")

"""
Computes the equation-of-state values (in eos_state) and sound speed.
"""
function update_eos(eos::AbstractEquationOfState, eos_state::AbstractEoSState, ρ, stress_work, dt)
    error("update_eos not implemented for $(typeof(eos))")
end

include("eos/tillotson_eos.jl") # Tillotson EoS implementation

# Strength Model
abstract type AbstractStrengthModel end
abstract type AbstractStrengthModelState end
init_strength_state(sm::AbstractStrengthModel) = error("init_strength_state not implemented for $(typeof(sm))")


function strength_model(strength_model::AbstractStrengthModel, strength_state::AbstractStrengthModelState, ρ, D_dev, W , dt)
    error("strength_model not implemented for $(typeof(strength_model))")
end

include("strength_models/elastic_strength.jl")  # Elastic strength model implementation


# ---------------------------------------------------------------------------- #
#                                Composite Types                               #
# ---------------------------------------------------------------------------- #
struct SolidMaterial{T, EoS<:AbstractEquationOfState, SM<:AbstractStrengthModel} <: AbstractMaterial
    eos::EoS
    strength_model::SM
    ρ::T    # Reference density for Body creation
end

struct SolidMaterialState{T, EoSState<:AbstractEoSState, SMState<:AbstractStrengthModelState} <: AbstractMaterialState
    c::T
    eos_state::EoSState
    strength_state::SMState
end


function get_initial_material_state(material::SolidMaterial{T}) where {T}
    # Initialize using model specific functions
    eos_state, c0 = init_eos_state(material.eos)
    strength_state = init_strength_state(material.strength_model)
    
    return SolidMaterialState(c0, eos_state, strength_state)
end


function get_soundspeed(material::SolidMaterial, mat_state::SolidMaterialState)
    return mat_state.c
end




# ---------------------------------------------------------------------------- #
#                         Material Model Implementation                        #
# ---------------------------------------------------------------------------- #
function material_model(
    material::SolidMaterial{T, EoS, SM}, 
    mat_state::SolidMaterialState{T, EoSState, SMState}, 
    F::SMatrix{3,3,T,9}, C::SMatrix{3,3,T,9}, V0::T, m::T, dt::T
) where {T, EoS<:AbstractEquationOfState, SM<:AbstractStrengthModel, EoSState<:AbstractEoSState, SMState<:AbstractStrengthModelState}
    # Kinematic part
    J = det(F)
    ρ = m / (J * V0)

    D = (0.5) * (C + C')
    W = (0.5) * (C - C')
    trD = tr(D)
    D_dev = D - (trD / T(3)) * one(SMatrix{3,3,T,9})

    # Strength model update
    s_new, strength_state_new, c_p = strength_model(material.strength_model, mat_state.strength_state, ρ, D_dev, W , dt)

    # Stress work
    p_old = mat_state.eos_state.p
    stress_work = (-p_old * trD + dot(s_new, D_dev)) / ρ

    # EOS update
    eos_state_new, c_eos = update_eos(material.eos, mat_state.eos_state, ρ, stress_work, dt)

    # get wavespeed
    c = sqrt(c_p^2 + c_eos^2)

    # Assemble the Cauchy stress tensor
    σ = s_new - eos_state_new.p * one(SMatrix{3,3,T,9})

    # Construct new material state
    mat_state_new = SolidMaterialState{T, EoSState, SMState}(c, eos_state_new, strength_state_new)

    return σ, mat_state_new
end





