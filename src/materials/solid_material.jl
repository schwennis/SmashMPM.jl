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
#   1. Kinematics           F = regularize_deformation(damage, damage_state, F)   (fractured: J^{1/3} I)
#                           kin = Kinematics(F, C, V0, m, dt)
#   2. Thermodynamics       th = thermo(eos, eos_state, e)
#   3. Elastic Predictor    s_tr, el_tr, G = elastic_trial(elasticity, elastic_state, kin)
#   4. Plastic Corrector    s, pl = return_map(plasticity, plastic_state, s_tr, G, th, kin)
#                           el = consistent_elastic_state(elasticity, el_tr, s, kin)
#   5. Damage               dm = update_damage(damage, damage_state, s, pl_st, th, kin, st.c)
#                           s = degradation(damage, dm) * s
#   6. Viscosity            q = viscous_pressure(viscosity, kin, st.c)   (0 if fractured and J > 1)
#   7. Energy Update (generic, per unit mass, ρ0 = m / V0)
#        heat = [½(τ_n + τ_{n+1}) : D dt - (W_el,{n+1} - W_el,n)] / ρ0
#        e    = e_n + heat - (½(p_n + p_{n+1}) + q) (J_{n+1} - J_n) / ρ0      (trapezoidal, implicit in p)
#   8. EoS                  p, c_b = update_eos(eos, eos_state, ρ, e)     (J > 1: c_b = c_0)
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

"""
    init_state(model_component, rng, n_p, V)

Initialize the state of one of the `n_p` particles of a body with volume `V`.
Overwrite this for components with randomly distributed initial states.
"""
init_state(model_component, rng, n_p, V) = init_state(model_component)


# Equation of State ----------------------------------------------------------- #
"""
    update_eos(eos::AbstractEquationOfState, eos_state::AbstractEoSState, ρ, e) -> (eos_state_write, c_bulk)
"""
@inline update_eos(eos::AbstractEquationOfState, eos_state::AbstractEoSState, ρ, e) = _not_implemented(:update_eos, eos)

"""
    update_eos_dpde(eos::AbstractEquationOfState, eos_state::AbstractEoSState, ρ, e) -> (eos_state_write, c_bulk, ∂p/∂e|_ρ)

Like `update_eos`, but additionally returns the derivative of the pressure with respect
to the specific internal energy at constant density. It is used to solve the implicit
trapezoidal energy update. Default: ∂p/∂e = 0 (exact for barotropic EoS like Murnaghan).
"""
@inline function update_eos_dpde(eos::AbstractEquationOfState, eos_state::AbstractEoSState, ρ, e)
    st, c = update_eos(eos, eos_state, ρ, e)
    return st, c, zero(e)
end

"""
    correct_eos_energy(eos, eos_state_pred, ρ, e_new, dpde, Δe) -> eos_state_write

EoS state at the corrected energy `e_new = e_pred + Δe`, starting from the state evaluated
at `e_pred` (density ρ unchanged). Default: exact re-evaluation with `update_eos`. EoS with
a cheap alternative overwrite it: linearization p + ∂p/∂e Δe (Tillotson) or reuse (Murnaghan).
"""
@inline correct_eos_energy(eos::AbstractEquationOfState, st, ρ, e_new, dpde, Δe) = first(update_eos(eos, st, ρ, e_new))

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



"""
    kirchhoff_deviator(el::AbstractElasticity, el_state, J) -> τ'
Returns:
- `τ'`: deviatoric Kirchhoff stress J s of the (undegraded) elastic state, `J = det F`.
Needed for the trapezoidal deviatoric work ½(τ_n + τ_{n+1}) : D dt.
"""
@inline kirchhoff_deviator(el::AbstractElasticity, el_state, J) = _not_implemented(:kirchhoff_deviator, el)

"""
    elastic_energy(el::AbstractElasticity, el_state, J) -> W_el
Returns:
- `W_el`: stored (undegraded) elastic shear energy per unit REFERENCE volume.
The plastic heat is the deviatoric work minus the change of this energy.
"""
@inline elastic_energy(el::AbstractElasticity, el_state, J) = _not_implemented(:elastic_energy, el)


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
    update_damage(damage::AbstractDamage, damage_state, s, pl_state, thermo, kin::Kinematics, c) -> damage_state_write
`c` is the longitudinal wave speed of the previous time step.
"""
@inline update_damage(damage::AbstractDamage, damage_state, s, pl_state, thermo, kin::Kinematics, c) = _not_implemented(:update_damage, damage)

"""
    degradation(damage::AbstractDamage, damage_state) -> g ∈ [0,1]
"""
@inline degradation(damage::AbstractDamage, damage_state) = _not_implemented(:degradation, damage)

"""
    degrade_pressure(damage::AbstractDamage, damage_state, p) -> p_degraded (Default: p)
"""
@inline degrade_pressure(damage::AbstractDamage, damage_state, p) = p

"""
    is_fractured(damage::AbstractDamage, damage_state) -> Bool (Default: false)
True if the particle counts as completely destroyed (no shear and tensile strength left).
Fractured particles get an isotropic F and no artificial viscosity in expansion (J > 1).
"""
@inline is_fractured(::AbstractDamage, damage_state) = false

"""
    fractured_max_J(damage::AbstractDamage) -> J_max (Default: 10)
Upper bound for J = det F of fractured particles.
"""
@inline fractured_max_J(::AbstractDamage) = 10

"""
    regularize_deformation(damage::AbstractDamage, damage_state, F) -> F_reg

For fractured particles F carries no information any more (no shear stress), but
F_{n+1} = (I + C dt) F_n keeps multiplying with the noisy velocity gradient. For isolated
fragments this grows exponentially until F is singular (det F = 0). Therefore F is
replaced by the isotropic tensor with the same (bounded) volume ratio:
    F_reg = J^{1/3} I,   J = clamp(det F, eps, J_max)
Intact particles are returned unchanged.
"""
@inline function regularize_deformation(damage::AbstractDamage, damage_state, F::SMatrix{3,3,T,9}) where {T}
    is_fractured(damage, damage_state) || return F
    J = clamp(det(F), eps(T), T(fractured_max_J(damage)))   # det F <= 0 only after a CFL violation
    return cbrt(J) * one(SMatrix{3,3,T,9})
end


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
@inline update_damage(::NoDamage, damage_state, s, pl_state, thermo, kin::Kinematics, c) = damage_state
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
    plasticity::PL = NullPlasticity()
    damage::DM = NoDamage()
    viscosity::AV = NoViscosity()
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

function initial_material_state(mat::SolidMaterial, rng, n_p, V)
    st = initial_material_state(mat)
    return SolidMaterialState(
        c             = st.c,
        e             = st.e,
        eos_state     = st.eos_state,
        elastic_state = st.elastic_state,
        plastic_state = init_state(mat.plasticity, rng, n_p, V),
        damage_state  = init_state(mat.damage, rng, n_p, V),
    )
end

@inline soundspeed(::SolidMaterial, mat_state::SolidMaterialState) = mat_state.c

# Same regularization as in material_model (state of t_n), used for the stored F
@inline regularize_deformation(mat::SolidMaterial, st::SolidMaterialState, F) =
    regularize_deformation(mat.damage, st.damage_state, F)



# ---------------------------------------------------------------------------- #
#                         Material Model implementation                        #
# ---------------------------------------------------------------------------- #
#   1. Kinematics           F = regularize_deformation(damage, damage_state, F)   (fractured: J^{1/3} I)
#                           kin = Kinematics(F, C, V0, m, dt)
#   2. Thermodynamics       th = thermo(eos, eos_state, e)
#   3. Elastic Predictor    s_tr, el_tr, G = elastic_trial(elasticity, elastic_state, kin)
#   4. Plastic Corrector    s, pl = return_map(plasticity, plastic_state, s_tr, G, th, kin)
#                           el = consistent_elastic_state(elasticity, el_tr, s, kin)
#   5. Damage               dm = update_damage(damage, damage_state, s, pl_st, th, kin, st.c)
#                           s = degradation(damage, dm) * s
#   6. Viscosity            q = viscous_pressure(viscosity, kin, st.c)   (0 if fractured and J > 1)
#   7. Energy Update (generic, per unit mass, ρ0 = m / V0)
#        heat = [½(τ_n + τ_{n+1}) : D dt - (W_el,{n+1} - W_el,n)] / ρ0
#        e    = e_n + heat - (½(p_n + p_{n+1}) + q) (J_{n+1} - J_n) / ρ0      (trapezoidal, implicit in p)
#   8. EoS                  p, c_b = update_eos(eos, eos_state, ρ, e)     (J > 1: c_b = c_0)
#   9. Assembly of σ        σ = s - (p + q) * I
@inline function material_model(mat::SolidMaterial, st::SolidMaterialState, F::SMatrix{3,3,T,9}, C::SMatrix{3,3,T,9}, V0::T, m::T, dt::T, dx::T) where T
    I3 = one(SMatrix{3,3,T,9})

    # 1. Kinematics
    # F = (I + C dt) F_n is the unregularized trial F, so J_n = det(F_n) is exact here.
    # Fractured particles (state of t_n) get F = J^{1/3} I with bounded J; the G2P2G
    # kernel stores the same regularized F.
    det_Δf = det(I3 + dt * C)
    J_n    = det_Δf > zero(T) ? det(F) / det_Δf : det(F)
    F_r    = regularize_deformation(mat.damage, st.damage_state, F)
    kin    = Kinematics(F_r, C, V0, m, dt, dx)
    J      = det(F_r)
    ρ0     = m / V0                     # reference density of the particle
    ρ      = ρ0 / J
    D      = (kin.C + kin.C') * T(0.5)

    # 2. Thermodynamics
    th  = thermo(mat.eos, st.eos_state, st.e)

    # 3. Elastic Predictor
    s_tr, el_tr, G = elastic_trial(mat.elasticity, st.elastic_state, kin)

    # 4. Plastic Corrector
    s, pl_st       = return_map(mat.plasticity, st.plastic_state, s_tr, G, th, kin)
    el_st          = consistent_elastic_state(mat.elasticity, el_tr, s, kin)

    # 5. Damage
    g_n   = degradation(mat.damage, st.damage_state)
    dm_st = update_damage(mat.damage, st.damage_state, s, pl_st, th, kin, st.c)
    g     = degradation(mat.damage, dm_st)
    s     = g * s

    # 6. Viscosity
    # In fractured, expanded material the (rate based, lagging) q only does negative
    # work during recompaction of the fragments and drives e below zero -> switched off.
    fractured_expanded = is_fractured(mat.damage, dm_st) && J > one(T)
    q = fractured_expanded ? zero(T) : T(viscous_pressure(mat.viscosity, kin, st.c))

    # 7. Energy Update (generic)
    # 7a. Heat from the deviatoric work (plasticity + damage):
    #     heat = ½(τ_n + τ_{n+1}) : Δε - ΔW_el,   Δε = D dt,   τ = J s (degraded)
    #     Trapezoidal work minus the change of the stored energy is consistent to second
    #     order. (W_trial - W_{n+1} would only be first order and creates energy.)
    τ_n   = g_n * kirchhoff_deviator(mat.elasticity, st.elastic_state, J_n)
    τ_np1 = J * s
    W_n   = g_n * elastic_energy(mat.elasticity, st.elastic_state, J_n)
    W_np1 = g   * elastic_energy(mat.elasticity, el_st, J)
    heat  = (T(0.5) * dt * dot(τ_n + τ_np1, D) - (W_np1 - W_n)) / ρ0

    # 7b. Volume work, trapezoidal in p:  Δv = 1/ρ_{n+1} - 1/ρ_n = (J - J_n) / ρ0
    #     e = e_n + heat - (½(p_n + p_{n+1}) + q) Δv,   p_{n+1} = p(ρ_{n+1}, e)
    #     Predictor with p_n, then one linearized correction p(e) ≈ p* + k (e - e*):
    #     e = e* - ½ (p* - p_n) Δv / (1 + ½ k Δv)
    Δv     = (J - J_n) / ρ0
    p_n    = degrade_pressure(mat.damage, st.damage_state, T(th.p))   # pressure that acted at t_n
    e_pred = st.e + heat - (p_n + q) * Δv

    eos_pred, c_b, dpde = update_eos_dpde(mat.eos, st.eos_state, ρ, e_pred)
    p_raw  = T(pressure(eos_pred))
    p_pred = T(degrade_pressure(mat.damage, dm_st, p_raw))
    k      = iszero(p_raw) ? T(dpde) : T(dpde) * (p_pred / p_raw)     # degradation also scales ∂p/∂e

    # Safeguard: the denominator approaches 0 only for extreme compression per step
    denom  = one(T) + T(0.5) * k * Δv
    denom  = isfinite(denom) ? max(denom, T(0.5)) : one(T)
    e      = e_pred - T(0.5) * (p_pred - p_n) * Δv / denom

    # 8. EoS at the corrected energy: NO second full evaluation. p_{n+1} = p* + ∂p/∂e (e - e*)
    #    is exactly the pressure that the implicit energy equation above was solved with, so e and
    #    the stored p stay consistent; c_b comes from the predictor (O(Δe) difference).
    eos_st = correct_eos_energy(mat.eos, eos_pred, ρ, e, T(dpde), e - e_pred)
    p = degrade_pressure(mat.damage, dm_st, pressure(eos_st))

    # In expansion the EoS sound speed is unreliable (cold expanded Tillotson branch,
    # Murnaghan cutoff) -> reference bulk sound speed c_0
    c_b = J > one(T) ? T(reference_soundspeed(mat.eos)) : c_b

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

include("plasticity/j2_plasticity.jl")

include("damage/grady_kipp_damage.jl")