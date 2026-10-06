"""
    Iron(::Type{T}=Float64; eos=:tillotson, elasticity=:hypoelastic,
         plasticity=NoPlasticity(), damage=NoDamage(), viscosity=NoViscosity())

Iron with selectable EoS and elasticity (symbols, parameters provided).
Plasticity, damage, and artificial viscosity are passed as ready-made instances, e.g.
`plasticity = J2Plasticity(; σ_y0 = 2.5e8, H = 1.0e9)`.
"""
function Iron(
    ::Type{T} = Float64;
    eos::Symbol = :tillotson,
    elasticity::Symbol = :hypoelastic,
    plasticity::AbstractPlasticity = NoPlasticity(),
    damage::AbstractDamage = NoDamage(),
    viscosity::AbstractArtificialViscosity = NoViscosity(),
) where {T<:AbstractFloat}

    ρ_ref = T(7874.0)   # kg/m^3

    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            ρ_ref,       # ρ0  (kg/m^3)
            T(128.0e9),  # A   (Pa)
            T(105.0e9),  # B   (Pa)
            T(5.0),      # α
            T(5.0),      # β
            T(9.5e6),    # E0  (J/kg)
            T(2.4e6),    # Eiv (J/kg)
            T(8.67e6),   # Ecv (J/kg)
            T(0.5),      # a
            T(0.15),     # b
        )
    elseif eos === :murnaghan
        MurnaghanEOS{T}(ρ_ref, T(113.5e9), T(5.32), T(0.9))
    else
        error("Unknown EoS for iron: :$eos")
    end

    μ = T(105e9)
    el_inst = if elasticity === :hyperelastic
        HyperElasticity{T}(μ)
    elseif elasticity === :hypoelastic
        HypoElasticity{T}(μ)
    else
        error("Unknown elasticity model for iron: :$elasticity")
    end

    return SolidMaterial(; eos = eos_inst, elasticity = el_inst,
                         plasticity, damage, viscosity, ρ0 = ρ_ref)
end