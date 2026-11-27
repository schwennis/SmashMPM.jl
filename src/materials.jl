# Materials
abstract type AbstractMaterial end
abstract type AbstractMaterialState end
struct NoMaterialState <: AbstractMaterialState end

# ---------------------------------------------------------------------------- #
#                                   Fallback                                   #
# ---------------------------------------------------------------------------- #
function material_model(material::AbstractMaterial, mat_state, F, C, V0, m, dt, dx)
    error("material_model not implemented for $(typeof(material))")
end


"""
    regularize_deformation(material, mat_state, F) -> F

Regularized deformation gradient that is stored on the particle after the update
(e.g. isotropic F for fractured particles). `mat_state` is the state of t_n.
Default: F unchanged.
"""
@inline regularize_deformation(::AbstractMaterial, mat_state, F) = F


"""
    initial_material_state(material, rng, n_p, V)

Initial state of one of the `n_p` particles of a body with volume `V`.
Defaults to the deterministic `initial_material_state(material)`.
"""
initial_material_state(material::AbstractMaterial, rng, n_p, V) = initial_material_state(material)

"""
    initial_material_states(material, rng, n_p, V) -> (material, states)

Initial states of ALL `n_p` particles of a body with volume `V`, used by
`build_mpm_model`. The returned material is the one of the particle set; it may
carry data that was sampled for this body (e.g. the Grady-Kipp flaws), so every
body gets its own instance. Default: `initial_material_state(material, rng, n_p, V)`
per particle and the unchanged material.
"""
initial_material_states(material::AbstractMaterial, rng, n_p, V) =
    (material, [initial_material_state(material, rng, n_p, V) for _ in 1:n_p])


# ---------------------------------------------------------------------------- #
#                     Internal energy (transfer heat bookkeeping)              #
# ---------------------------------------------------------------------------- #
"""
    has_internal_energy(material) -> Bool

True if the material state carries a specific internal energy that can take up
the heat of the transfer (`APIC(heating=true)` / `PolyPIC(heating=true)`).
"""
has_internal_energy(::AbstractMaterial) = false

"""
    add_heat(material, mat_state, Δe) -> mat_state

Adds the specific energy `Δe` (per unit mass) to the internal energy of the state.
Default: the state is returned unchanged (the heat is lost).
"""
@inline add_heat(::AbstractMaterial, mat_state, Δe) = mat_state

"""
    internal_energy(material, mat_state) -> e

Specific internal energy of the state (0 for materials without internal energy).
"""
@inline internal_energy(::AbstractMaterial, mat_state) = false   # type neutral 0

"""
    strain_energy(material, mat_state, F) -> W

Stored elastic energy per unit REFERENCE volume (0 if not implemented).
Only used for the energy diagnostics (`energy_budget`).
"""
@inline strain_energy(::AbstractMaterial, mat_state, F) = false


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