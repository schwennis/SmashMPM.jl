"""
    Iron(::Type{T}=Float64; eos=:tillotson, elasticity=:hypoelastic, damage=:no_damage,
         plasticity=NullPlasticity(), viscosity=NoViscosity())

Iron with selectable EoS, elasticity and damage (symbols, parameters provided).
`damage` also accepts a ready-made instance. Plasticity and artificial viscosity are
passed as ready-made instances, e.g. `plasticity = J2Plasticity(; σ_y0 = 2.5e8, H = 1.0e9)`.
"""
function Iron(
    ::Type{T} = Float64;
    eos::Symbol = :tillotson,
    elasticity::Symbol = :hypoelastic,
    damage::Union{Symbol, AbstractDamage} = :no_damage,
    plasticity::AbstractPlasticity = NullPlasticity(),
    viscosity::AbstractArtificialViscosity = NoViscosity(),
) where {T<:AbstractFloat}

    ρ_ref = T(7874.0)   # kg/m^3

    # ---------------------------------------------------------------------------- #
    #                               Equation of State                              #
    # ---------------------------------------------------------------------------- #
    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            ρ0 = ρ_ref,
            A = T(128.0e9),
            B = T(105.0e9),
            α = T(5.0),
            β = T(5.0),
            E0 = T(9.5e6),
            Eiv = T(2.4e6),
            Ecv = T(8.67e6),
            a = T(0.5),
            b = T(0.15),
            ρiv = zero(T)
        )
    elseif eos === :murnaghan
        MurnaghanEOS{T}(ρ_ref, T(113.5e9), T(5.32), T(0.9))
    else
        error("Unknown EoS for iron: :$eos")
    end

    # ---------------------------------------------------------------------------- #
    #                                  Elasticity                                  #
    # ---------------------------------------------------------------------------- #
    μ = T(105e9)
    el_inst = if elasticity === :hyperelastic
        HyperElasticity{T}(μ)
    elseif elasticity === :hypoelastic
        HypoElasticity{T}(μ)
    else
        error("Unknown elasticity model for iron: :$elasticity")
    end

    # ---------------------------------------------------------------------------- #
    #                                    Damage                                    #
    # ---------------------------------------------------------------------------- #
    damage_inst = if damage isa AbstractDamage
        damage
    elseif damage === :grady_kipp
        GradyKippDamage{T}(
            E     = T(211.0e9),
            m     = 9.0,
            k     = 1.0e31,           # σ_t = 350 MPa at V_ref = 1 cm^3
            n_tot = 1.0e10,
        )
    elseif damage === :no_damage
        NoDamage()
    else
        error("Unknown damage model for iron: :$damage")
    end

    return SolidMaterial(; eos = eos_inst, elasticity = el_inst,
                         plasticity, damage = damage_inst, viscosity, ρ = ρ_ref)
end