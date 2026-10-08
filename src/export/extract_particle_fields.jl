# ---------------------------------------------------------------------------- #
#               Particle data (incl. material state) for HDF5 export           #
# ---------------------------------------------------------------------------- #
# One small method per state type, no reflection / metaprogramming.
# Scalars (p, e, eps_p, damage) are read through the accessor functions of the
# material pipeline (pressure, equivalent_plastic_strain, damage_variable), so a
# new EoS, plasticity or damage model needs NO extra line here.
# Only tensor-valued elastic states need ONE extra method in _elastic_fields below
# (unknown types are skipped with a warning).
#
# Datasets (names are paths inside the HDF5 file):
#   mat_state/c                       (N,)    longitudinal wave speed
#   mat_state/e                       (N,)    specific internal energy
#   mat_state/eps_p                   (N,)    equivalent plastic strain (0 without plasticity)
#   mat_state/damage                  (N,)    damage variable D (0 without damage)
#   mat_state/eos_state/p             (N,)    pressure
#   mat_state/elastic_state/s         (N,9)   deviatoric stress    (HypoElasticity)
#   mat_state/elastic_state/bbar_e    (N,9)   elastic isochoric left Cauchy-Green b̄ₑ (HyperElasticity)
#
# Changes compared to the old SolidMaterial export:
#   - eos_state/e (Tillotson) is now mat_state/e and exists for every EoS
#   - strength_state/{s,b} is now elastic_state/{s,bbar_e}
#
# (F and affine are written by the exporter itself.)
# 3x3 matrices are stored flat and row-major, like F and affine:
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

# ---------------------------------------------------------------------------- #
#                       Collect over all particle sets                         #
# ---------------------------------------------------------------------------- #
"All mat_state/... fields of one CPU particle set."
_set_fields(particles) = _state_fields(particles.mat_state)

_fill_value(::Type{E}) where {E} = E <: AbstractFloat ? E(NaN) : zero(E)

"""
Merge the fields of all particle sets into arrays over all `total` particles, in
the same order as the flat datasets position/mass/... Sets that do not own a
field (e.g. NeoHookean has no eos_state, a hypoelastic set has no `bbar_e`) get NaN.
"""
function _gather_particle_fields(particle_sets, total::Int)
    per_set = [_set_fields(ps.particles) for ps in particle_sets]
    names   = sort!(unique(reduce(vcat, [collect(keys(d)) for d in per_set]; init = String[])))

    merged = Dict{String, Array}()
    for name in names
        arrays = [d[name] for d in per_set if haskey(d, name)]
        E      = promote_type(eltype.(arrays)...)
        lead   = size(first(arrays))[1:end-1]
        out    = fill(_fill_value(E), lead..., total)

        offset = 1
        for (ps, d) in zip(particle_sets, per_set)
            N = length(ps.particles.mass)
            if haskey(d, name) && N > 0
                selectdim(out, ndims(out), offset:(offset + N - 1)) .= d[name]
            end
            offset += N
        end
        merged[name] = out
    end
    return merged
end

# ---------------------------------------------------------------------------- #
#                                  HDF5 / XDMF                                 #
# ---------------------------------------------------------------------------- #
function _write_dataset(file, name::String, arr::AbstractArray, compression_level::Int)
    if compression_level > 0
        chunk = (size(arr)[1:end-1]..., min(size(arr)[end], 1024))
        file[name, chunk = chunk, compress = compression_level] = arr
    else
        file[name] = arr
    end
end

function _xdmf_particle_attributes(h5_filename::String, fields::AbstractDict, total::Int)
    io = IOBuffer()
    for name in sort!(collect(keys(fields)))
        arr   = fields[name]
        lead  = size(arr)[1:end-1]
        atype = isempty(lead) ? "Scalar" : "Matrix"      # scalars or flat 3x3 (9 comps), same as F/affine
        dims  = join((total, lead...), " ")
        print(io, """
              <Attribute Name="$(replace(name, "/" => "."))" AttributeType="$(atype)" Center="Node">
                <DataItem Dimensions="$(dims)" NumberType="Float" Precision="$(sizeof(eltype(arr)))" Format="HDF">
                  $(h5_filename):/$(name)
                </DataItem>
              </Attribute>
        """)
    end
    return String(take!(io))
end