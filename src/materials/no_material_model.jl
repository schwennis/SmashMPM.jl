# ---------------------------------------------------------------------------- #
#                                NoMaterialModel                               #
# ---------------------------------------------------------------------------- #
struct NoMaterialModel<:AbstractMaterial end

function get_initial_material_state(::NoMaterialModel)
    return NoMaterialState()
end

# No material model, returns identity stress and does not update the material state
@inline function material_model(material::NoMaterialModel, mat_state::NoMaterialState, F, C, V0, m, dt::T) where T
    σ = one(SMatrix{3, 3, T, 9})

    return σ, mat_state
end

function get_soundspeed(material::NoMaterialModel, material_cache::NoMaterialState)
    return 0
end