# Materials
abstract type AbstractMaterial end
abstract type AbstractMaterialState end

"""
    add_heat(mat_state, Δe) -> mat_state

Add the specific energy Δe (J/kg) booked by the particle-grid transfer to the internal
energy of the state. States without an internal energy ignore it.
"""
@inline add_heat(st::AbstractMaterialState, Δe) = st
struct NoMaterialState <: AbstractMaterialState end

# ---------------------------------------------------------------------------- #
#                                   Fallback                                   #
# ---------------------------------------------------------------------------- #
function material_model(material::AbstractMaterial, mat_state, F, C, V0, m, dt, dx)
    error("material_model not implemented for $(typeof(material))")
end


"""
    initial_material_state(material, rng, n_p, V)

Initial state of one of the `n_p` particles of a body with volume `V`.
Defaults to the deterministic `initial_material_state(material)`.
"""
initial_material_state(material::AbstractMaterial, rng, n_p, V) = initial_material_state(material)


# ---------------------------------------------------------------------------- #
#                        Material Model Implementations                        #
# ---------------------------------------------------------------------------- #
include("materials/hyperelastic.jl")
include("materials/no_material_model.jl")
include("materials/solid_material.jl")


# ---------------------------------------------------------------------------- #
#                               Material Configs                               #
# ---------------------------------------------------------------------------- #
include("materials/material_configs/iron.jl")
include("materials/material_configs/basalt.jl")
include("materials/material_configs/aluminum.jl")
include("materials/material_configs/nylon.jl")
