# ---------------------------------------------------------------------------- #
#               Particle data (incl. material state) for HDF5 export           #
# ---------------------------------------------------------------------------- #
# Everything here works on ONE CPU particle set. The exporter writes every set
# into its own group `particles/ParticleSet{i}/`, so a set only carries the fields
# its material really has (no NaN padding for fields of other materials).
#
# Material state: one small method per state type, no reflection / metaprogramming.
# Scalars (p, e, eps_p, damage) are read through the accessor functions of the
# material pipeline (pressure, equivalent_plastic_strain, damage_variable), so a
# new EoS, plasticity or damage model needs NO extra line here.
# Only tensor-valued elastic states need ONE extra method in _elastic_fields below
# (unknown types are skipped with a warning).
#
# Datasets of the material state (relative to `particles/ParticleSet{i}/`):
#   mat_state/c                       (N,)    longitudinal wave speed
#   mat_state/e                       (N,)    specific internal energy
#   mat_state/eps_p                   (N,)    equivalent plastic strain (0 without plasticity)
#   mat_state/damage                  (N,)    damage variable D (0 without damage)
#   mat_state/eos_state/p             (N,)    pressure
#   mat_state/elastic_state/s         (N,9)   deviatoric stress    (HypoElasticity)
#   mat_state/elastic_state/bbar_e    (N,9)   elastic isochoric left Cauchy-Green b̄ₑ (HyperElasticity)
#
# Stress (see `export_stress` below):
#   stress                            (N,9)   Cauchy stress σ
#   von_mises                         (N,)    von Mises equivalent stress sqrt(3/2 s:s)
#
# 3x3 matrices are stored flat and row-major:
#   [A11 A12 A13 A21 A22 A23 A31 A32 A33]  -> Julia array (9, N)

# ---------------------------------------------------------------------------- #
#                                    Helpers                                   #
# ---------------------------------------------------------------------------- #
"Vector of 3x3 matrices -> (9, N) array, row-major per particle."
function _matrices(M::AbstractVector{<:SMatrix{3, 3, T}}) where {T}
    A = Array{T}(undef, 9, length(M))
    for n in eachindex(M), r in 1:3, c in 1:3
        A[3*(r-1) + c, n] = M[n][r, c]
    end
    return A
end

"Vector of 3-vectors -> (3, N) array."
function _vectors(v::AbstractVector{<:SVector{3, T}}) where {T}
    A = Array{T}(undef, 3, length(v))
    for n in eachindex(v), d in 1:3
        A[d, n] = v[n][d]
    end
    return A
end

# ---------------------------------------------------------------------------- #
#                                    Stress                                    #
# ---------------------------------------------------------------------------- #
# The solver does not store σ on the particles (it only lives inside the G2P2G
# kernel). It is recomputed here from the stored state by calling the material
# model with a zero time step and zero velocity gradient:
#   - every rate term vanishes (dt * ...), so the elastic, plastic, damage and
#     energy states are returned unchanged
#   - the EoS is evaluated at the current density and energy -> same p as stored
#   - the artificial viscosity q (needs the velocity gradient) is 0
# The result is the Cauchy stress σ = g s - p I of the stored state, WITHOUT the
# artificial bulk viscosity. No extra method is needed for new material
# components, as long as their update is consistent for dt = 0. Otherwise add a
# method of `export_stress` for that material.
"""
    export_stress(material, mat_state, F, V0, m, dx) -> σ

Cauchy stress of one particle, recomputed from its stored state for the export
(without artificial viscosity). Default: `material_model` with `dt = 0`, `C = 0`.
"""
@inline function export_stress(material::AbstractMaterial, mat_state, F::SMatrix{3, 3, T, 9},
                               V0::T, m::T, dx::T) where {T}
    C = zero(SMatrix{3, 3, T, 9})
    σ, _ = material_model(material, mat_state, F, C, V0, m, zero(T), dx)
    return SMatrix{3, 3, T, 9}(σ)
end

"Cauchy stress of all particles of one CPU particle set."
function _particle_stress(p_set, dx::T) where {T}
    p   = p_set.particles
    mat = p_set.material
    N   = length(p.mass)
    σ   = Vector{SMatrix{3, 3, T, 9}}(undef, N)
    Threads.@threads for n in 1:N
        σ[n] = export_stress(mat, p.mat_state[n], p.F[n], p.V0[n], p.mass[n], dx)
    end
    return σ
end

"von Mises equivalent stress sqrt(3/2 s:s) with s = dev(σ)."
@inline function von_mises(σ::SMatrix{3, 3, T, 9}) where {T}
    s = σ - (tr(σ) / 3) * one(SMatrix{3, 3, T, 9})
    return sqrt(T(1.5) * dot(s, s))
end

# ---------------------------------------------------------------------------- #
#                         Fields per state type (dispatch)                     #
# ---------------------------------------------------------------------------- #
# --- equation of state (generic via accessor) ------------------------------- #
_eos_fields(s::AbstractVector{<:AbstractEoSState}) = Dict{String, Array}(
    "p" => [pressure(x) for x in s],
)

# --- elasticity ------------------------------------------------------------- #
_elastic_fields(s::AbstractVector{<:HypoElasticState}) = Dict{String, Array}(
    "s" => _matrices([x.s for x in s]),
)
_elastic_fields(s::AbstractVector{<:HyperElasticState}) = Dict{String, Array}(
    "bbar_e" => _matrices([x.bbar_e for x in s]),
)

# --- material state --------------------------------------------------------- #
_state_fields(::AbstractVector{NoMaterialState}) = Dict{String, Array}()

function _state_fields(ms::AbstractVector{<:SolidMaterialState{T}}) where {T}
    fields = Dict{String, Array}(
        "mat_state/c"      => [s.c for s in ms],
        "mat_state/e"      => [s.e for s in ms],
        # accessors return `false` for NullState → converted to 0
        "mat_state/eps_p"  => [T(equivalent_plastic_strain(s.plastic_state)) for s in ms],
        "mat_state/damage" => [T(max(damage_variable(s.plastic_state),
                                     damage_variable(s.damage_state))) for s in ms],
    )
    for (name, arr) in _eos_fields([s.eos_state for s in ms])
        fields["mat_state/eos_state/" * name] = arr
    end
    for (name, arr) in _elastic_fields([s.elastic_state for s in ms])
        fields["mat_state/elastic_state/" * name] = arr
    end
    return fields
end

# Fallback: do not crash the simulation because of an unknown state type
_elastic_fields(s::AbstractVector) =
    (@warn "HDF5 export: no fields defined for elastic state $(eltype(s))" maxlog=1; Dict{String, Array}())
_state_fields(s::AbstractVector) =
    (@warn "HDF5 export: no fields defined for material state $(eltype(s))" maxlog=1; Dict{String, Array}())

"All mat_state/... fields of one CPU particle set."
_mat_state_fields(particles) = _state_fields(particles.mat_state)

# ---------------------------------------------------------------------------- #
#                        All selected fields of one set                        #
# ---------------------------------------------------------------------------- #
"""
    _particle_set_fields(sel, p_set, set_idx, grid, spline, dx) -> Vector{Pair{String, Array}}

Datasets of one CPU particle set, in a fixed order, restricted to the selection
`sel` (see `HDF5Exporter`). `position` is always first, it is the geometry.
Expensive fields (velocity/affine via G2P, stress) are only computed if selected.
"""
function _particle_set_fields(sel::FieldSelection, p_set, set_idx::Int, grid, spline, dx::T) where {T}
    p = p_set.particles
    N = length(p.mass)
    out = Pair{String, Array}[]

    position = Matrix{T}(undef, 3, N)
    position[1, :] .= p.pos.x
    position[2, :] .= p.pos.y
    position[3, :] .= p.pos.z
    push!(out, "position" => position)

    if wants(sel, :velocity) || wants(sel, :affine)
        v_p, affines = extract_velocities(grid, p_set, spline)
        wants(sel, :velocity) && push!(out, "velocity" => _vectors(v_p))
        wants(sel, :affine)   && push!(out, "affine"   => _matrices(affines))
    end
    wants(sel, :F)          && push!(out, "F"          => _matrices(p.F))
    wants(sel, :mass)       && push!(out, "mass"       => Vector{T}(p.mass))
    wants(sel, :volume)     && push!(out, "volume"     => Vector{T}(p.V0))
    wants(sel, :wave_speed) && push!(out, "wave_speed" => Vector{T}(p.wave_speed))
    wants(sel, :id)         && push!(out, "id"         => Vector{Int}(p.id))
    wants(sel, :set_index)  && push!(out, "set_index"  => fill(set_idx, N))

    if wants(sel, :stress) || wants(sel, :von_mises)
        σ = _particle_stress(p_set, dx)
        wants(sel, :stress)    && push!(out, "stress"    => _matrices(σ))
        wants(sel, :von_mises) && push!(out, "von_mises" => von_mises.(σ))
    end

    if wants_mat_state(sel)
        ms = _mat_state_fields(p)
        for name in sort!(collect(keys(ms)))
            wants_mat_state(sel, name) && push!(out, name => ms[name])
        end
    end
    return out
end