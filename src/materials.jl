abstract type AbstractMaterial end
abstract type AbstractMaterialState end
struct NoMaterialState <: AbstractMaterialState end

# ---------------------------------------------------------------------------- #
#                                   Fallback                                   #
# ---------------------------------------------------------------------------- #
function material_model(material::AbstractMaterial, mat_cache, F, C, V0, m, dt)
    error("material_model not implemented for $(typeof(material))")
end


# ---------------------------------------------------------------------------- #
#                        Material Model Implementations                        #
# ---------------------------------------------------------------------------- #
include("materials/hyperelastic.jl")
include("materials/no_material_model.jl")