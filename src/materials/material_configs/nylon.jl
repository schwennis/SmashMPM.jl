"""
    Nylon(::Type{T}=Float64; eos=:tillotson, elasticity=:hypoelastic,
          plasticity=J2Plasticity(σ_y0=T(10e6), H=T(2.19e6)), damage=:no_damage,
          viscosity=NoViscosity())

Nylon projectile as in Benz and Asphaug (1994) and Yan et al. (2026), Table 1.
`damage` accepts a symbol or a ready-made instance. Plasticity and artificial viscosity
are passed as ready-made instances.
"""
function Nylon(
    ::Type{T} = Float64;
    eos::Symbol = :tillotson,
    elasticity::Symbol = :hypoelastic,
    damage::Union{Symbol, AbstractDamage} = :no_damage,
    plasticity::AbstractPlasticity = J2Plasticity(σ_y0 = T(10e6), H = T(2.19e6)),
    viscosity::AbstractArtificialViscosity = NoViscosity(),
) where {T<:AbstractFloat}

    ρ_ref = T(1180.0)   # kg/m^3

# ---------------------------------------------------------------------------- #
#                               Equation of State                              #
# ---------------------------------------------------------------------------- #
    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            ρ_ref,
            T(10.1e9),   # A   (Pa)
            T(33.8e9),   # B   (Pa)
            T(10.0),     # α
            T(5.0),      # β
            T(7.0e6),    # E0  (J/kg)
            T(2.0e6),    # Eiv (J/kg)
            T(2.4e6),    # Ecv (J/kg)
            T(0.6),      # a
            T(2.0),      # b
        )
    else
        error("Unknown EoS for nylon: :$eos")
    end

# ---------------------------------------------------------------------------- #
#                                  Elasticity                                  #
# ---------------------------------------------------------------------------- #
    μ = T(219e6 / (2 * (1 + 0.45)))   # E = 219 MPa, ν = 0.45
    el_inst = if elasticity === :hyperelastic
        HyperElasticity{T}(μ)
    elseif elasticity === :hypoelastic
        HypoElasticity{T}(μ)
    else
        error("Unknown elasticity model for nylon: :$elasticity")
    end

# ---------------------------------------------------------------------------- #
#                                    Damage                                    #
# ---------------------------------------------------------------------------- #
    damage_inst = if damage isa AbstractDamage
        damage
    elseif damage === :no_damage
        NoDamage()
    else
        error("Unknown damage model for nylon: :$damage")
    end

    return SolidMaterial(; eos = eos_inst, elasticity = el_inst,
                         plasticity, damage = damage_inst, viscosity, ρ = ρ_ref)
end