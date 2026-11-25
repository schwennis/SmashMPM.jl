# ---------------------------------------------------------------------------- #
#                                 HDF5 Exporter                                #
# ---------------------------------------------------------------------------- #
# File layout (3-vectors / 3x3-matrices are stored flat, row-major, i.e. a
# Julia array of size (3, N) / (9, N), which is (N, 3) / (N, 9) in HDF5):
#
#   particles/ParticleSet{i}/position     (3, N)  always written (geometry)
#   particles/ParticleSet{i}/velocity     (3, N)  G2P of the read grid state
#   particles/ParticleSet{i}/affine       (9, N)  APIC matrix C (G2P)
#   particles/ParticleSet{i}/F            (9, N)  deformation gradient
#   particles/ParticleSet{i}/mass, volume, wave_speed      (N,)
#   particles/ParticleSet{i}/id           (N,)    global particle id
#   particles/ParticleSet{i}/set_index    (N,)    = i (handy after merging blocks)
#   particles/ParticleSet{i}/stress       (9, N)  Cauchy stress σ (without artificial viscosity)
#   particles/ParticleSet{i}/von_mises    (N,)    von Mises stress (off by default)
#   particles/ParticleSet{i}/mat_state/...        (see extract_particle_fields.jl)
#
#   grid/position     (3, M)  node positions, always written (geometry)
#   grid/node_index   (3, M)  global node indices (i, j, k), 1-based, may be <= 0
#   grid/mass         (M,)
#   grid/momentum     (3, M)
#   grid/force        (3, M)
#   grid/velocity     (3, M)  momentum / mass, 0 for empty nodes (off by default)
#   grid/origin, grid/dx      grid frame (origin = position of node (1, 1, 1))
#
# Every particle set has its own group; empty sets are skipped. In the XDMF file
# every set is its own <Grid Name="ParticleSet{i}"> inside one spatial
# collection, so ParaView shows one block per particle set (+ "MPM_Grid").
# Use the "XDMF Reader" or "Xdmf3 Reader T" in ParaView. ("Xdmf3 Reader S"
# merges all grids of a spatial collection into the pieces of a single block.)
#
# The grid is a flat list of nodes, just like the particles. It is taken from the
# read state of the grid (the same state `extract_velocities` uses):
#   DenseGrid:  all nodes
#   SparseGrid: all nodes of the active tiles only (nodes outside of a BoxDomain
#               are left out). No conversion to a dense grid is needed.
#
# Attributes: root: time, step
#             particles: n_particles, n_sets
#             particles/ParticleSet{i}: n_particles, set_index, material
#             grid: type, n_nodes (+ n_tiles, tile_size for the SparseGrid)

# ---------------------------------------------------------------------------- #
#                                Field selection                               #
# ---------------------------------------------------------------------------- #
"Particle fields that can be selected (`position` is always written)."
const PARTICLE_FIELDS = (:velocity, :affine, :F, :mass, :volume, :wave_speed, :id,
                         :set_index, :stress, :von_mises, :mat_state)
"Particle fields written by default."
const DEFAULT_PARTICLE_FIELDS = (:velocity, :affine, :F, :mass, :volume, :wave_speed, :id,
                                 :set_index, :stress, :mat_state)

"Grid fields that can be selected (`position` is always written)."
const GRID_FIELDS = (:node_index, :mass, :momentum, :force, :velocity)
"Grid fields written by default."
const DEFAULT_GRID_FIELDS = (:node_index, :mass, :momentum, :force)

"""
Parsed field selection. `names` are the plain fields, `mat_state` the selected
material state paths ("mat_state" = everything, "mat_state/eos_state" = all
fields below that group, "mat_state/e" = exactly that field).
"""
struct FieldSelection
    names::Set{Symbol}
    mat_state::Vector{String}
end

@inline wants(sel::FieldSelection, name::Symbol) = name in sel.names
@inline wants_mat_state(sel::FieldSelection) = !isempty(sel.mat_state)
@inline wants_mat_state(sel::FieldSelection, path::String) =
    any(p -> path == p || startswith(path, p * "/"), sel.mat_state)

_as_list(spec::Union{Symbol, AbstractString}) = (spec,)
_as_list(spec) = spec

"""
    FieldSelection(spec, available, defaults; what, allow_mat_state)

`spec` is a collection of Symbols/Strings (or a single one). The entries `:all`
and `:default` expand to all / the default fields, `:position` is ignored (it is
always written). Entries starting with `mat_state` select material state fields.
Unknown names throw an `ArgumentError` when the exporter is created, not in the
middle of a simulation.
"""
function FieldSelection(spec, available::Tuple, defaults::Tuple;
                        what::String, allow_mat_state::Bool)
    names     = Set{Symbol}()
    mat_state = String[]

    function add!(entry)
        s = rstrip(String(entry), '/')
        if s == "all"
            foreach(add!, available)
        elseif s == "default"
            foreach(add!, defaults)
        elseif s == "position"
            # geometry, always written
        elseif allow_mat_state && (s == "mat_state" || startswith(s, "mat_state/"))
            s in mat_state || push!(mat_state, s)
        elseif Symbol(s) in available
            push!(names, Symbol(s))
        else
            throw(ArgumentError("HDF5Exporter: unknown $(what) field `$(s)`. " *
                                "Available: $(join(available, ", ")) (or :all, :default)."))
        end
        return nothing
    end
    foreach(add!, _as_list(spec))
    return FieldSelection(names, mat_state)
end


# ---------------------------------------------------------------------------- #
#                                   Exporter                                   #
# ---------------------------------------------------------------------------- #
"""
    HDF5Exporter(; output_dir = "output", filename_prefix = "sim_", write_xdmf = true,
                   compression_level = 3, export_particles = true, export_grid = false,
                   particle_fields = DEFAULT_PARTICLE_FIELDS,
                   grid_fields = DEFAULT_GRID_FIELDS)

Writes one HDF5 file (+ an XDMF file for ParaView) per output step. Every particle
set is written into its own group `particles/ParticleSet{i}` and is its own block
in ParaView.

`particle_fields` / `grid_fields` select the written quantities, as a tuple or
vector of Symbols (or Strings). Position is always written.

Particle fields: `:velocity, :affine, :F, :mass, :volume, :wave_speed, :id,
:set_index, :stress, :von_mises, :mat_state`. Single material state fields can be
selected by their path, e.g. `"mat_state/eos_state/p"` or `"mat_state/elastic_state"`.

Grid fields: `:node_index, :mass, :momentum, :force, :velocity`.

`:all` and `:default` expand to all / the default fields.

# Examples
```julia
HDF5Exporter(particle_fields = (:velocity, :stress, "mat_state/eos_state/p"))
HDF5Exporter(particle_fields = (:default, :von_mises))     # defaults + von Mises
HDF5Exporter(particle_fields = :all, export_grid = true, grid_fields = (:mass, :velocity))
```
"""
struct HDF5Exporter <: AbstractExporter
    output_dir::String
    filename_prefix::String
    write_xdmf::Bool
    compression_level::Int
    export_particles::Bool
    export_grid::Bool
    particle_fields::FieldSelection
    grid_fields::FieldSelection
end

function HDF5Exporter(; output_dir::AbstractString = "output",
                        filename_prefix::AbstractString = "sim_",
                        write_xdmf::Bool = true,
                        compression_level::Integer = 3,
                        export_particles::Bool = true,
                        export_grid::Bool = false,
                        particle_fields = DEFAULT_PARTICLE_FIELDS,
                        grid_fields = DEFAULT_GRID_FIELDS)
    p_sel = FieldSelection(particle_fields, PARTICLE_FIELDS, DEFAULT_PARTICLE_FIELDS;
                           what = "particle", allow_mat_state = true)
    g_sel = FieldSelection(grid_fields, GRID_FIELDS, DEFAULT_GRID_FIELDS;
                           what = "grid", allow_mat_state = false)
    return HDF5Exporter(String(output_dir), String(filename_prefix), write_xdmf,
                        Int(compression_level), export_particles, export_grid, p_sel, g_sel)
end


# ---------------------------------------------------------------------------- #
#                                  Grid nodes                                  #
# ---------------------------------------------------------------------------- #
_trim_nodes(A::AbstractMatrix, m::Int) = size(A, 2) == m ? A : A[:, 1:m]
_trim_nodes(A::AbstractVector, m::Int) = length(A) == m ? A : A[1:m]

"""
    _export_grid_nodes(grid) -> NamedTuple

Flat node list of the grid (read state) with the fields `node_index`, `position`,
`mass`, `momentum`, `force`. For a `SparseGrid` only the nodes of the active
tiles are collected, sorted by tile key (reproducible order).
"""
function _export_grid_nodes(grid::DenseGrid)
    state = grid.state_read
    T     = eltype(state.mass)
    dims  = size(state.mass)
    M     = prod(dims)

    ijk      = Matrix{Int}(undef, 3, M)
    position = Matrix{T}(undef, 3, M)
    for (n, I) in enumerate(CartesianIndices(dims))     # i runs fastest, like vec(array)
        ijk[1, n], ijk[2, n], ijk[3, n] = I[1], I[2], I[3]
        x = node_position(grid.origin, grid.inv_dx, I[1], I[2], I[3])
        position[1, n], position[2, n], position[3, n] = x[1], x[2], x[3]
    end

    momentum = Matrix{T}(undef, 3, M)
    momentum[1, :] .= vec(Array(state.momentum.x))
    momentum[2, :] .= vec(Array(state.momentum.y))
    momentum[3, :] .= vec(Array(state.momentum.z))

    force = Matrix{T}(undef, 3, M)
    force[1, :] .= vec(Array(state.force.x))
    force[2, :] .= vec(Array(state.force.y))
    force[3, :] .= vec(Array(state.force.z))

    return (; node_index = ijk, position,
              mass       = vec(Array(state.mass)),
              momentum,
              force)
end

function _export_grid_nodes(grid::SparseGrid)
    state = grid.state_read
    T     = eltype(state.mass)
    n     = grid.n_active_read            # only tiles 1:n_active_read are in use

    tile_keys = Array(view(grid.map_read.tile_keys, 1:n))
    order     = sortperm(tile_keys)

    mass       = Array(view(state.mass,       :, 1:n))
    px         = Array(view(state.momentum.x, :, 1:n))
    py         = Array(view(state.momentum.y, :, 1:n))
    pz         = Array(view(state.momentum.z, :, 1:n))
    fx         = Array(view(state.force.x,    :, 1:n))
    fy         = Array(view(state.force.y,    :, 1:n))
    fz         = Array(view(state.force.z,    :, 1:n))

    M_max    = TILE_VOLUME * n
    ijk      = Matrix{Int}(undef, 3, M_max)
    position = Matrix{T}(undef, 3, M_max)
    momentum = Matrix{T}(undef, 3, M_max)
    force    = Matrix{T}(undef, 3, M_max)
    out_mass = Vector{T}(undef, M_max)

    m = 0
    for t in order, l in 1:TILE_VOLUME
        i, j, k = node_index(tile_keys[t], l)
        node_in_domain(grid.domain, i, j, k) || continue    # boundary tiles may stick out of a box
        m += 1

        ijk[1, m], ijk[2, m], ijk[3, m] = i, j, k
        x = node_position(grid.origin, grid.inv_dx, i, j, k)
        position[1, m], position[2, m], position[3, m] = x[1], x[2], x[3]

        out_mass[m] = mass[l, t]
        force[1, m], force[2, m], force[3, m] = fx[l, t], fy[l, t], fz[l, t]
        momentum[1, m], momentum[2, m], momentum[3, m] = px[l, t], py[l, t], pz[l, t]
    end

    return (; node_index = _trim_nodes(ijk, m),
              position   = _trim_nodes(position, m),
              mass       = _trim_nodes(out_mass, m),
              momentum   = _trim_nodes(momentum, m),
              force      = _trim_nodes(force, m))
end

"Nodal velocity momentum / mass, 0 for (numerically) empty nodes."
function _grid_velocity(mass::AbstractVector{T}, momentum::AbstractMatrix{T}) where {T}
    v = zeros(T, size(momentum))
    cutoff = sqrt(floatmin(T))
    for n in eachindex(mass)
        mass[n] > cutoff || continue
        for d in 1:3
            v[d, n] = momentum[d, n] / mass[n]
        end
    end
    return v
end

"Selected grid datasets, `position` first."
function _grid_fields(sel::FieldSelection, nodes::NamedTuple)
    out = Pair{String, Array}["position" => nodes.position]
    wants(sel, :node_index) && push!(out, "node_index" => nodes.node_index)
    wants(sel, :mass)       && push!(out, "mass"       => nodes.mass)
    wants(sel, :momentum)   && push!(out, "momentum"   => nodes.momentum)
    wants(sel, :force)      && push!(out, "force"      => nodes.force)
    wants(sel, :velocity)   && push!(out, "velocity"   => _grid_velocity(nodes.mass, nodes.momentum))
    return out
end

# Grid frame and type-specific metadata
function _write_grid_info!(file, grid::AbstractGrid, n_nodes::Int)
    T = typeof(grid.inv_dx)
    file["grid/origin"] = collect(grid.origin)
    file["grid/dx"]     = one(T) / grid.inv_dx

    g = file["grid"]
    attributes(g)["type"]    = string(nameof(typeof(grid)))
    attributes(g)["n_nodes"] = n_nodes
    _write_grid_tile_info!(g, grid)
    return nothing
end

_write_grid_tile_info!(g, ::DenseGrid) = nothing
function _write_grid_tile_info!(g, grid::SparseGrid)
    attributes(g)["n_tiles"]   = grid.n_active_read
    attributes(g)["tile_size"] = TILE_SIZE
    return nothing
end


# ---------------------------------------------------------------------------- #
#                                    Output                                    #
# ---------------------------------------------------------------------------- #
"""
One point cloud of the output: a group in the HDF5 file and a <Grid> (= one
ParaView block) in the XDMF file. `fields[1]` is the position.
"""
struct _ExportBlock
    name::String                        # XDMF / ParaView block name
    group::String                       # HDF5 group
    n::Int                              # number of points
    fields::Vector{Pair{String, Array}} # dataset name (relative to group) => data
end

function _write_dataset(file, name::String, arr::AbstractArray, compression_level::Int)
    if compression_level > 0
        chunk = (size(arr)[1:end-1]..., min(size(arr)[end], 1024))
        file[name, chunk = chunk, compress = compression_level] = arr
    else
        file[name] = arr
    end
end

function _write_block!(file, block::_ExportBlock, compression_level::Int)
    for (name, arr) in block.fields
        _write_dataset(file, "$(block.group)/$(name)", arr, compression_level)
    end
    return nothing
end

function write_output(exporter::HDF5Exporter, model::MPMModel, step::Int, time::Real = model.t)
    mkpath(exporter.output_dir)

    cpu_model = model_to_cpu(model)
    grid      = cpu_model.grid
    T         = eltype(grid.origin)
    dx        = one(T) / T(grid.inv_dx)
    level     = exporter.compression_level

    padded_idx  = Printf.@sprintf("%06d", step)
    h5_filename = "$(exporter.filename_prefix)$(padded_idx).h5"
    h5_path     = joinpath(exporter.output_dir, h5_filename)

    # 1. Particles: one block per (non-empty) particle set
    particle_blocks = _ExportBlock[]
    set_info        = Tuple{Int, String}[]          # (set index, material name) per block
    if exporter.export_particles
        for (i, p_set) in enumerate(cpu_model.particle_sets)
            N = length(p_set.particles.mass)
            N == 0 && continue
            fields = _particle_set_fields(exporter.particle_fields, p_set, i, grid,
                                          cpu_model.shapefunction, dx)
            push!(particle_blocks, _ExportBlock("ParticleSet$(i)", "particles/ParticleSet$(i)", N, fields))
            push!(set_info, (i, string(nameof(typeof(p_set.material)))))
        end
    end
    n_particles = sum(b -> b.n, particle_blocks; init = 0)

    # 2. Grid nodes: no conversion to a dense grid, a SparseGrid exports its active tiles
    grid_block = nothing
    if exporter.export_grid
        nodes = _export_grid_nodes(grid)
        M     = size(nodes.position, 2)
        if M > 0
            grid_block = _ExportBlock("MPM_Grid", "grid", M, _grid_fields(exporter.grid_fields, nodes))
        end
    end

    h5open(h5_path, "w") do file
        for (block, (set_idx, material)) in zip(particle_blocks, set_info)
            _write_block!(file, block, level)
            a = attributes(file[block.group])
            a["n_particles"] = block.n
            a["set_index"]   = set_idx
            a["material"]    = material
        end
        if !isempty(particle_blocks)
            attributes(file["particles"])["n_particles"] = n_particles
            attributes(file["particles"])["n_sets"]      = length(particle_blocks)
        end

        if !isnothing(grid_block)
            _write_block!(file, grid_block, level)
            _write_grid_info!(file, grid, grid_block.n)
        end

        attributes(file)["time"] = Float64(time)
        attributes(file)["step"] = step
    end

    if exporter.write_xdmf
        blocks = isnothing(grid_block) ? particle_blocks : [particle_blocks; grid_block]
        _write_xdmf_metadata(exporter.output_dir, exporter.filename_prefix, padded_idx,
                             blocks, Float64(time))
    end

    return h5_path
end


# ---------------------------------------------------------------------------- #
#                                     XDMF                                     #
# ---------------------------------------------------------------------------- #
# Display names in ParaView; everything else (mat_state/...) gets "/" -> "."
const _XDMF_NAMES = Dict(
    "velocity"   => "Velocity",   "affine"     => "Affine",    "F"        => "F",
    "mass"       => "Mass",       "volume"     => "Volume",    "wave_speed" => "WaveSpeed",
    "id"         => "ID",         "set_index"  => "SetIndex",  "stress"   => "Stress",
    "von_mises"  => "VonMises",   "momentum"   => "Momentum",  "force"    => "Force",
    "node_index" => "NodeIndex",
)
_xdmf_name(name::String) = get(_XDMF_NAMES, name, replace(name, "/" => "."))

# Attribute type from the number of components (leading dimension)
function _xdmf_attribute_type(arr::AbstractArray)
    ndims(arr) == 1 && return "Scalar"
    ncomp = prod(size(arr)[1:end-1])
    ncomp == 3 && return "Vector"
    ncomp == 9 && return "Tensor"       # flat 3x3, row-major
    ncomp == 6 && return "Tensor6"
    return "Matrix"
end

_xdmf_number_type(::Type{<:AbstractFloat}) = "Float"
_xdmf_number_type(::Type{<:Signed})        = "Int"
_xdmf_number_type(::Type{<:Unsigned})      = "UInt"
_xdmf_number_type(::Type{Bool})            = "UChar"

"HDF5 / XDMF dimensions: Julia (c, N) -> \"N c\""
_xdmf_dims(arr::AbstractArray) = join(reverse(size(arr)), " ")

function _xdmf_data_item(io, h5_filename, path, arr, indent)
    E = eltype(arr)
    print(io, indent, """<DataItem Dimensions="$(_xdmf_dims(arr))" NumberType="$(_xdmf_number_type(E))" """,
          """Precision="$(sizeof(E))" Format="HDF">""", h5_filename, ":/", path, "</DataItem>\n")
end

function _xdmf_block(io, h5_filename::String, block::_ExportBlock)
    pos_name, pos = first(block.fields)
    print(io, """    <Grid Name="$(block.name)" GridType="Uniform">\n""")
    print(io, """      <Topology TopologyType="Polyvertex" NumberOfElements="$(block.n)"/>\n""")
    print(io, """      <Geometry GeometryType="XYZ">\n""")
    _xdmf_data_item(io, h5_filename, "$(block.group)/$(pos_name)", pos, "        ")
    print(io, """      </Geometry>\n""")
    for (name, arr) in block.fields[2:end]
        print(io, """      <Attribute Name="$(_xdmf_name(name))" AttributeType="$(_xdmf_attribute_type(arr))" Center="Node">\n""")
        _xdmf_data_item(io, h5_filename, "$(block.group)/$(name)", arr, "        ")
        print(io, """      </Attribute>\n""")
    end
    print(io, """    </Grid>\n""")
end

# Generates the companion XDMF XML wrapper for ParaView ingestion. All blocks
# (particle sets, grid) are point clouds (Polyvertex) in one spatial collection.
function _write_xdmf_metadata(output_dir::String, prefix::String, padded_idx::String,
                              blocks::AbstractVector{_ExportBlock}, time::Float64)
    h5_filename = "$(prefix)$(padded_idx).h5"
    xmf_path    = joinpath(output_dir, "$(prefix)$(padded_idx).xmf")

    io = IOBuffer()
    print(io, """<?xml version="1.0" encoding="utf-8"?>\n""")
    print(io, """<Xdmf Version="3.0">\n<Domain>\n""")
    print(io, """  <Grid Name="SimulationState" GridType="Collection" CollectionType="Spatial">\n""")
    print(io, """    <Time Value="$(time)"/>\n""")
    for block in blocks
        _xdmf_block(io, h5_filename, block)
    end
    print(io, "  </Grid>\n</Domain>\n</Xdmf>\n")

    write(xmf_path, take!(io))
    return xmf_path
end