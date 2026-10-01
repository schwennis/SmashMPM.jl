"""
    Basalt(::Type{T}=Float64; eos=:tillotson, strength=:elastic, damage=:none)

Creates a Basalt material with the specified equation of state (EoS), strength model, and damage model (not implemented yet).
"""
function Basalt(
    ::Type{T} = Float64; 
    eos::Symbol = :tillotson, 
    strength::Symbol = :hypoelastic, 
    # damage::Symbol = :none
) where {T<:AbstractFloat}

    ρ_ref = T(2860.0)  # Reference density for basalt in kg/m^3

    # equation of state
    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            ρ_ref,   # ρ0  (kg/m^3)
            T(26.7e9),   # A   (Pa)
            T(26.7e9),   # B   (Pa)
            T(5.0),      # α
            T(5.0),      # β
            T(487.0e6),  # E0  (J/kg)
            T(4.72e6),   # Eiv (J/kg)
            T(18.2e6),   # Ecv (J/kg)
            T(0.5),      # a
            T(1.5)       # b
        )
    elseif eos === :murnaghan
        MurnaghanEOS{T}(
            ρ_ref,  # ρ0  (kg/m^3)
            T(26.7e9),  # K0  (Pa)
            T(5.5),     # n
            T(0.9)      # η_limit (relative compression)
        )
    else
        error("Unknown EoS for basalt: :$eos")
    end

    # strength model 
    strength_inst = if strength === :hyperelastic
        HyperElasticStrengthModel{T}(T(2.27e10))   # μ = 22.7 GPa
    elseif strength === :hypoelastic
        HypoElasticStrengthModel{T}(T(2.27e10))   # μ = 22.7 GPa
    else
        error("Unknown strength model for basalt: :$strength")
    end

    # damage model (not implemented yet)
    # damage_inst = if damage === :none
    #     NoDamageModel()
    # else
    #     error("Unknown damage model for basalt: :$damage")
    # end

    return SolidMaterial(eos_inst, strength_inst, ρ_ref)  # ρ = 2860.0 kg/m^3
end