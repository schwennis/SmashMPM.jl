"""
    Aluminum(::Type{T}=Float64; eos=:tillotson, elasticity=:hypoelastic, damage=:no_damage,
             plasticity=NullPlasticity(), viscosity=NoViscosity())

Aluminum with selectable EoS, elasticity and damage (symbols, parameters provided).
`damage` also accepts a ready-made instance. Plasticity and artificial viscosity are
passed as ready-made instances, e.g. `plasticity = J2Plasticity(; σ_y0 = 2.76e8, H = 6.8e8)`.
"""
function Aluminum(
    ::Type{T} = Float64;
    eos::Symbol = :tillotson,
    elasticity::Symbol = :hypoelastic,
    damage::Union{Symbol, AbstractDamage} = :no_damage,
    plasticity::AbstractPlasticity = NullPlasticity(),
    viscosity::AbstractArtificialViscosity = NoViscosity(),
) where {T<:AbstractFloat}

    ρ_ref = T(2700.0)   # kg/m^3

    # ---------------------------------------------------------------------------- #
    #                               Equation of State                              #
    # ---------------------------------------------------------------------------- #
    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            ρ0 = ρ_ref,
            A = T(75.2e9),
            B = T(65.0e9),
            α = T(5.0),
            β = T(5.0),
            E0 = T(5e6),
            Eiv = T(3e6),
            Ecv = T(13.9e6),
            a = T(0.5),
            b = T(1.63),
            ρiv = zero(T)
        )
    else
        error("Unknown EoS for aluminum: :$eos")
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
        error("Unknown elasticity model for aluminum: :$elasticity")
    end

    # ---------------------------------------------------------------------------- #
    #                                    Damage                                    #
    # ---------------------------------------------------------------------------- #
    damage_inst = if damage isa AbstractDamage
        damage
    elseif damage === :grady_kipp
        GradyKippDamage{T}(
            E     = T(69.0e9),
            m     = 9.0,
            k     = 1.3e27,           # σ_t = 310 MPa at V_ref = 1 cm^3
            n_tot = 1.0e10,
        )
    elseif damage === :no_damage
        NoDamage()
    else
        error("Unknown damage model for aluminum: :$damage")
    end

    return SolidMaterial(; eos = eos_inst, elasticity = el_inst,
                         plasticity, damage = damage_inst, viscosity, ρ = ρ_ref)
end