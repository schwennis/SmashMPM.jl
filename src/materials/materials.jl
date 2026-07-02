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
include("NeoHookean.jl")
include("LinearElastic.jl")
include("no_material_model.jl")