function Aluminum(
    ::Type{T} = Float64;
    eos::Symbol = :tillotson,
    elasticity::Symbol = :hypoelastic,
    plasticity::AbstractPlasticity = NullPlasticity(),
    damage::AbstractDamage = NoDamage(),
    viscosity::AbstractArtificialViscosity = NoViscosity(),
) where T

    ρ_ref = T(2700.0)   # kg/m^3

    # ---------------------------------------------------------------------------- #
    #                               Equation of State                              #
    # ---------------------------------------------------------------------------- #
    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            ρ_ref,      # ρ  (kg/m^3)
            T(75.2e9),  # A   (Pa)
            T(65.0e9),  # B   (Pa)
            T(5.0),     # α
            T(5.0),     # β
            T(5e6),     # E0  (J/kg)
            T(3e6),     # Eiv (J/kg)
            T(13.9e6),  # Ecv (J/kg)
            T(0.5),     # a
            T(1.63),    # b
        )
    else
        error("Unknown EoS for aluminum: $eos")
    end

    # ---------------------------------------------------------------------------- #
    #                                  Elasticity                                  #
    # ---------------------------------------------------------------------------- #
    μ = T(26e9)
    el_inst = if elasticity === :hyperelastic
        HyperElasticity{T}(μ)
    elseif elasticity === :hypoelastic
        HypoElasticity{T}(μ)
    else
        error("Unknown elasticity model for aluminium: :$elasticity")
    end

    return SolidMaterial(; eos = eos_inst, elasticity = el_inst,
                         plasticity, damage, viscosity, ρ = ρ_ref)
end