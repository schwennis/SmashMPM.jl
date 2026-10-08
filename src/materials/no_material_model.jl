# ---------------------------------------------------------------------------- #
#                                NoMaterialModel                               #
# ---------------------------------------------------------------------------- #
struct NoMaterialModel<:AbstractMaterial end

function initial_material_state(::NoMaterialModel)
    return NoMaterialState()
end

#
@inline function material_model(material::NoMaterialModel, mat_state::NoMaterialState, F, C, V0, m, dt::T, dx::T) where T
    σ = zero(SMatrix{3, 3, T, 9})

    return σ, mat_state
end

function soundspeed(material::NoMaterialModel, mat_state::NoMaterialState)
    return false    # type neutral 0
end