"""
    Iron(::Type{T}=Float64; eos=:tillotson, strength=:elastic, damage=:none)

Creates an Iron material with the specified equation of state (EoS), strength model, and damage model (not implemented yet).
"""
function Iron(
    ::Type{T} = Float64; 
    eos::Symbol = :tillotson, 
    strength::Symbol = :elastic, 
    # damage::Symbol = :none
) where {T<:AbstractFloat}

    # 1. Equation of State (EoS)
    eos_inst = if eos === :tillotson
        TillotsonEOS{T}(
            T(7874.0),   # ρ0  (kg/m^3)
            T(128.0e9),  # A   (Pa)
            T(105.0e9),  # B   (Pa)
            T(5.0),      # α
            T(5.0),      # β
            T(9.5e6),    # E0  (J/kg)
            T(2.4e6),    # Eiv (J/kg)
            T(8.67e6),    # Ecv (J/kg)
            T(0.5),      # a
            T(0.15)      # b
        )
    elseif eos === :murnaghan
        MurnaghanEOS{T}(
            T(7874.0),   # ρ0  (kg/m^3)
            T(113.5e9),  # K0  (Pa)
            T(5.32),     # n
            T(0.9)       # η_limit (relative compression)
        )
    else
        error("Unknown EoS for iron: :$eos")
    end

    # strength model 
    strength_inst = if strength === :elastic
        ElasticStrengthModel{T}(T(105e9)) # μ = 105 GPa
    else
        error("Unknown strength model for iron: :$strength")
    end

    # # damage model
    # damage_inst = if damage === :none
    #     NoDamageModel()
    # # elseif damage === :johnson_cook_damage
    # #     JohnsonCookDamage{T}(...)
    # else
    #     error("Unknown damage model for iron: :$damage")
    # end

    return SolidMaterial(eos_inst, strength_inst, T(7800.0))
end