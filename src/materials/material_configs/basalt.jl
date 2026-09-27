"""
    Basalt(::Type{T}=Float64; eos=:tillotson, strength=:elastic, damage=:none)

Creates a Basalt material with the specified equation of state (EoS), strength model, and damage model (not implemented yet).
"""
function Basalt(
    ::Type{T} = Float64; 
    eos::Symbol = :tillotson, 
    strength::Symbol = :elastic, 
    # damage::Symbol = :none
) where {T<:AbstractFloat}

    # equation of state
    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            T(2700.0),   # ρ0  (kg/m^3)
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
    else
        error("Unbekannte EoS für Basalt: :$eos")
    end

    # strength model 
    strength_inst = if strength === :elastic
        ElasticStrengthModel{T}(T(2.3087e10))   # μ = 23.087 GPa
    else
        error("Unbekanntes Strength-Modell für Basalt: :$strength")
    end

    # damage model (not implemented yet)
    # damage_inst = if damage === :none
    #     NoDamageModel()
    # else
    #     error("Unbekanntes Schadensmodell für Basalt: :$damage")
    # end

    return SolidMaterial(eos_inst, strength_inst)
end