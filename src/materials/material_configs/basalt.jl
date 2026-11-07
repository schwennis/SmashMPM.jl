"""
    Basalt(::Type{T}=Float64; eos=:tillotson, elasticity=:hypoelastic, damage=:grady_kipp,
           plasticity=NullPlasticity(), viscosity=NoViscosity())

Basalt with selectable EoS, elasticity and damage (symbols, parameters provided).
`damage` also accepts a ready-made instance. Plasticity and artificial viscosity are
passed as ready-made instances, e.g. `plasticity = CollinsPlasticity(; Y0 = …, …)`.
"""
function Basalt(
    ::Type{T} = Float64;
    eos::Symbol = :tillotson,
    elasticity::Symbol = :hypoelastic,
    damage::Union{Symbol, AbstractDamage} = :grady_kipp,
    plasticity::AbstractPlasticity = NullPlasticity(),
    viscosity::AbstractArtificialViscosity = NoViscosity(),
) where {T<:AbstractFloat}

    ρ_ref = T(2860.0)   # kg/m^3

    # ---------------------------------------------------------------------------- #
    #                               Equation of State                              #
    # ---------------------------------------------------------------------------- #
    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            ρ_ref,
            T(26.7e9),   # A   (Pa)
            T(26.7e9),   # B   (Pa)
            T(5.0),      # α
            T(5.0),      # β
            T(487.0e6),  # E0  (J/kg)
            T(4.72e6),   # Eiv (J/kg)
            T(18.2e6),   # Ecv (J/kg)
            T(0.5),      # a
            T(1.5),      # b
        )
    elseif eos === :murnaghan
        MurnaghanEOS{T}(ρ_ref, T(26.7e9), T(5.5), T(0.9))
    else
        error("Unknown EoS for basalt: :$eos")
    end

    # ---------------------------------------------------------------------------- #
    #                                  Elasticity                                  #
    # ---------------------------------------------------------------------------- #
    μ = T(2.27e10)
    el_inst = if elasticity === :hyperelastic
        HyperElasticity{T}(μ)
    elseif elasticity === :hypoelastic
        HypoElasticity{T}(μ)
    else
        error("Unknown elasticity model for basalt: :$elasticity")
    end

    # ---------------------------------------------------------------------------- #
    #                                    Damage                                    #
    # ---------------------------------------------------------------------------- #
    damage_inst = if damage isa AbstractDamage
        damage
    elseif damage === :grady_kipp
        GradyKippDamage{T}(
            E     = T(53.1e9),
            m     = 8.5,
            k     = 3.0e39,
            n_tot = 1.0e9,
        )
    elseif damage === :no_damage
        NoDamage()
    else
        error("Unknown damage model for basalt: :$damage")
    end

    return SolidMaterial(; eos = eos_inst, elasticity = el_inst,
                         plasticity, damage = damage_inst, viscosity, ρ = ρ_ref)
end