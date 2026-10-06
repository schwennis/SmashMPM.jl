"""
    Basalt(::Type{T}=Float64; eos=:tillotson, elasticity=:hypoelastic,
           plasticity=NoPlasticity(), damage=NoDamage(), viscosity=NoViscosity())

Basalt with selectable EoS and elasticity (symbols, parameters provided).
Plasticity, damage, and artificial viscosity are passed as ready-made instances, e.g.
`plasticity = CollinsPlasticity(; Y0 = …, …)`.
"""
function Basalt(
    ::Type{T} = Float64;
    eos::Symbol = :tillotson,
    elasticity::Symbol = :hypoelastic,
    plasticity::AbstractPlasticity = NoPlasticity(),
    damage::AbstractDamage = NoDamage(),
    viscosity::AbstractArtificialViscosity = NoViscosity(),
) where {T<:AbstractFloat}

    ρ_ref = T(2860.0)   # kg/m^3

    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            ρ_ref,       # ρ0  (kg/m^3)
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

    μ = T(2.27e10)   # 22.7 GPa
    el_inst = if elasticity === :hyperelastic
        HyperElasticity{T}(μ)
    elseif elasticity === :hypoelastic
        HypoElasticity{T}(μ)
    else
        error("Unknown elasticity model for basalt: :$elasticity")
    end

    return SolidMaterial(; eos = eos_inst, elasticity = el_inst,
                         plasticity, damage, viscosity, ρ0 = ρ_ref)
end