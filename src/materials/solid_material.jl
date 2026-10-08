# ---------------------------------------------------------------------------- #
#                            Modular Solid Material                            #
# ---------------------------------------------------------------------------- #
#
# Here, the material model consists of a set of modular components:
#   Component       Type                        
#   EOS             ::AbstractEquationOfState   
#   Elasticity      ::AbstractElasticity        
#   Plasticity      ::AbstractPlasticity        
#   Damage          ::AbstractDamage            
#   Viscosity       ::AbstractViscosity
#
# The material model does the following steps every time step, using the follwing structs
#   0. material_model call with signature
#       mat::SolidMaterial, st::SolidMaterialState, F, C, V0, m, dt
#
#   1. Kinematics           kin = Kinematics(F, C, V0, m, dt)
#   2. Thermodynamics       th = thermo(eos, eos_state, e)
#   3. Elastic Predictor    s_tr, el_tr, G = elastic_trial(elasticity, elastic_state, kin)
#   4. Plastic Corrector    s, pl = return_map(plasticity, plastic_state, s_tr, G, th, kin)
#                           el = consistent_elastic_state(elasticity, el_tr, s, kin)
#   5. Damage               dm = update_damage(damage, damage_state, s, pl_st, th, kin)
#                           s = degradation(damage, dm) * s
#   6. Viscosity            q = viscous_pressure(viscosity, kin, st.c)
#   7. Energy Update (generic)  e += dt/ρ * (s : D' - (p_n + q) * tr(D))
#   8. EoS                  p, c_b = update_eos(eos, eos_state, ρ, e)
#   9. Assembly of σ        σ = s - (p + q) * I
#
#
# This is done so new models can be implemented via the functions concerning its slots.
# Coupling is done via accessors to prevent crashes due to different implementations of variables:
#   - pressure(eos_state)
#   - equivalent_plastic_strain(plastic_state)
#   - damage_variable(damage_state)
#   - thermo(eos, st, e)
# ---------------------------------------------------------------------------- #

# Abstract Types
abstract type AbstractEquationOfState end
abstract type AbstractEoSState end

abstract type AbstractElasticity end

abstract type AbstractPlasticity end

abstract type AbstractDamage end

abstract type AbstractArtificialViscosity end


# Empty Singleton stateless models
struct NullState end



# ---------------------------------------------------------------------------- #
#                                  Kinematics                                  #
# ---------------------------------------------------------------------------- #
struct Kinematics{T}
    F::SMatrix{3, 3, T, 9}
    C::SMatrix{3, 3, T, 9}
    V0::T
    m::T
    dt::T
    dx::T
end


# ---------------------------------------------------------------------------- #
#                           Interfaces and fallbacks                           #
# ---------------------------------------------------------------------------- #
# "not implemented" fallback for unimplemented functions
_not_implemented(f::Symbol, x) = error("$(f) is not implemented for $(typeof(x))")

# State initialisation ------------------------------------------------------- #
"""
    init_state(model_component)

Initialize the state of a material component.

# Arguments
- `model_component`: The material component for which to initialize the state.

# Returns
- The initialized state of the material component.
"""
@inline init_state(model_component) = _not_implemented(:init_state, model_component)


# Equation of State ----------------------------------------------------------- #
"""
    update_eos(eos::AbstractEquationOfState, eos_state::AbstractEoSState, ρ, e) -> (eos_state_write, c_bulk)
"""
@inline update_eos(eos::AbstractEquationOfState, eos_state::AbstractEoSState, ρ, e) = _not_implemented(:update_eos, eos)

"""
    reference_soundspeed(eos::AbstractEquationOfState) -> c_bulk in reference configuration
"""
@inline reference_soundspeed(eos::AbstractEquationOfState) = _not_implemented(:reference_soundspeed, eos)

"""
    pressure(eos_state::AbstractEoSState) -> p
Overwrite this function if the eos state has no field `p`
"""
@inline pressure(eos_state::AbstractEoSState) = eos_state.p

"""
    thermo(eos::AbstractEquationOfState, eos_state::AbstractEoSState, e) -> (p, T)
Thermodynamics interface for EoS. If EoS has temparature, overwrite this function to return the temperature. 
Otherwise, return zero.
"""
@inline thermo(::AbstractEquationOfState, eos_state::AbstractEoSState, e) = (p = pressure(eos_state), T = zero(e))


# Elasticity --------------------------------------------------------------- #
"""
    elastic_trial(el::AbstractElasticity, el_state, kin::Kinematics) -> (s_tr, el_tr, G)
Performs a trial update of the elastic state.
Returns:
- `s_tr`: trial stress
- `el_tr`: trial elastic state
- `G`: effective shear modulus for return mapping
"""
@inline elastic_trial(el::AbstractElasticity, el_state, kin::Kinematics) = _not_implemented(:elastic_trial, el)

"""
    consistent_elastic_state(el::AbstractElasticity, el_tr, s, kin::Kinematics) -> el_state
Performs a consistent update of the elastic state after plastic correction.
Returns:
- `el_state`: the consistent elastic state.
"""
@inline consistent_elastic_state(el::AbstractElasticity, el_tr, s, kin::Kinematics) = _not_implemented(:consistent_elastic_state, el)

"""
    shear_modulus(el::AbstractElasticity) -> G0
Returns:
- `G0`: shear modulus in the reference configuration.
"""
@inline shear_modulus(el::AbstractElasticity) = _not_implemented(:shear_modulus, el)



# Plasticity --------------------------------------------------------------- #
"""
    return_map(pl::AbstractPlasticity, pl_state, s_tr, G, thermo, kin::Kinematics) -> s, pl_state_write
Performs the return mapping algorithm for plasticity.
Returns:
- `s`: the updated stress after plastic correction.
- `pl_state_write`: the updated plastic state.
"""
@inline return_map(pl::AbstractPlasticity, pl_state, s_tr, G, thermo, kin::Kinematics) = _not_implemented(:return_map, pl)

# State accessors for plasticity, false if not implemented
@inline equivalent_plastic_strain(::NullState) = false
@inline damage_variable(::NullState) = false


# Damage --------------------------------------------------------------- #
"""
    update_damage(damage::AbstractDamage, damage_state, s, pl_state, thermo, kin::Kinematics) -> damage_state_write
"""
@inline update_damage(damage::AbstractDamage, damage_state, s, pl_state, thermo, kin::Kinematics) = _not_implemented(:update_damage, damage)

"""
    degradation(damage::AbstractDamage, damage_state) -> g ∈ [0,1]
"""
@inline degradation(damage::AbstractDamage, damage_state) = _not_implemented(:degradation, damage)

"""
    degrade_pressure(damage::AbstractDamage, damage_state, p) -> p_degraded (Default: p)
"""
@inline degrade_pressure(damage::AbstractDamage, damage_state, p) = p


# Artificial Viscosity --------------------------------------------------------------- #
"""
    viscous_pressure(viscosity::AbstractArtificialViscosity, kin::Kinematics, c) -> q
"""
@inline viscous_pressure(viscosity::AbstractArtificialViscosity, kin::Kinematics, c) = _not_implemented(:viscous_pressure, viscosity)



# ---------------------------------------------------------------------------- #
#                                  Null models                                 #
# ---------------------------------------------------------------------------- #
struct NullPlasticity <: AbstractPlasticity end
@inline init_state(::NullPlasticity) = NullState()
@inline return_map(::NullPlasticity, pl_state, s_tr, G, thermo, kin::Kinematics) = (s_tr, pl_state)


struct NoDamage <: AbstractDamage end
@inline init_state(::NoDamage) = NullState()
@inline update_damage(::NoDamage, damage_state, s, pl_state, thermo, kin::Kinematics) = damage_state
@inline degradation(::NoDamage, damage_state) = true    # true acts like type neutral 1

struct NoViscosity <: AbstractArtificialViscosity end
@inline viscous_pressure(::NoViscosity, kin::Kinematics, c) = zero(c)



# ---------------------------------------------------------------------------- #
#                                Composite types                               #
# ---------------------------------------------------------------------------- #
@kwdef struct SolidMaterial{T,
                            EoS<:AbstractEquationOfState,
                            EL<:AbstractElasticity,
                            PL<:AbstractPlasticity,
                            DM<:AbstractDamage,
                            AV<:AbstractArtificialViscosity} <: AbstractMaterial
    eos::EoS
    elasticity::EL
    plasticity::PL
    damage::DM
    viscosity::AV
    ρ::T
end

@kwdef struct SolidMaterialState{T,
                            EoSState<:AbstractEoSState,
                            ELState,
                            PLState,
                            DMState} <: AbstractMaterialState
    c::T
    e::T
    eos_state::EoSState
    elastic_state::ELState
    plastic_state::PLState
    damage_state::DMState
end


@inline function initial_material_state(mat::SolidMaterial{T}) where {T}
    c_b = reference_soundspeed(mat.eos)
    G0  = shear_modulus(mat.elasticity)
    c0  = sqrt(c_b^2 + 4G0 / (3mat.ρ))
    return SolidMaterialState(
        c             = c0,
        e             = zero(T),
        eos_state     = init_state(mat.eos),
        elastic_state = init_state(mat.elasticity),
        plastic_state = init_state(mat.plasticity),
        damage_state  = init_state(mat.damage),
    )
end

@inline soundspeed(::SolidMaterial, mat_state::SolidMaterialState) = mat_state.c



# ---------------------------------------------------------------------------- #
#                         Material Model implementation                        #
# ---------------------------------------------------------------------------- #
#   1. Kinematics           kin = Kinematics(F, C, V0, m, dt)
#   2. Thermodynamics       th = thermo(eos, eos_state, e)
#   3. Elastic Predictor    s_tr, el_tr, G = elastic_trial(elasticity, elastic_state, kin)
#   4. Plastic Corrector    s, pl = return_map(plasticity, plastic_state, s_tr, G, th, kin)
#                           el = consistent_elastic_state(elasticity, el_tr, s, kin)
#   5. Damage               dm = update_damage(damage, damage_state, s, pl_st, th, kin)
#                           s = degradation(damage, dm) * s
#   6. Viscosity            q = viscous_pressure(viscosity, kin, st.c)
#   7. Energy Update (generic)  e += dt/ρ * (s : D' - (p_n + q) * tr(D))
#   8. EoS                  p, c_b = update_eos(eos, eos_state, ρ, e)
#   9. Assembly of σ        σ = s - (p + q) * I
@inline function material_model(mat::SolidMaterial, st::SolidMaterialState, F::SMatrix{3,3,T,9}, C::SMatrix{3,3,T,9}, V0::T, m::T, dt::T, dx::T) where T
    # 1. Kinematics
    kin = Kinematics(F, C, V0, m, dt, dx)
    ρ   = kin.m / (det(kin.F) * kin.V0)
    D   = (kin.C + kin.C') * T(0.5)

    # 2. Thermodynamics
    th  = thermo(mat.eos, st.eos_state, st.e)

    # 3. Elastic Predictor
    s_tr, el_tr, G = elastic_trial(mat.elasticity, st.elastic_state, kin)

    # 4. Plastic Corrector
    s, pl_st       = return_map(mat.plasticity, st.plastic_state, s_tr, G, th, kin)
    el_st          = consistent_elastic_state(mat.elasticity, el_tr, s, kin)

    # 5. Damage
    dm_st = update_damage(mat.damage, st.damage_state, s, pl_st, th, kin)
    g     = degradation(mat.damage, dm_st)
    s     = g * s

    # 6. Viscosity
    q = viscous_pressure(mat.viscosity, kin, st.c)

    # 7. Energy Update (generic)
    e = st.e + dt / ρ * (dot(s, D) - (th.p + q) * tr(D))

    # 8. EoS
    eos_st, c_b = update_eos(mat.eos, st.eos_state, ρ, e)
    p = degrade_pressure(mat.damage, dm_st, pressure(eos_st))

    # Total sound speed
    c = sqrt(c_b^2 + g * 4G / (3ρ))

    # 9. Assembly of σ
    σ = s - (p + q) * I

    return σ, SolidMaterialState(c = c, e = e, eos_state = eos_st,
                                 elastic_state = el_st, plastic_state = pl_st,
                                 damage_state = dm_st)
end



# ---------------------------------------------------------------------------- #
#                             Model Implementations                            #
# ---------------------------------------------------------------------------- #
include("eos/murnaghan_eos.jl")
include("eos/tillotson_eos.jl")

include("elasticity/hyperelasticity.jl")
include("elasticity/hypoelasticity.jl")

include("viscosity/bulk_viscosity.jl")
