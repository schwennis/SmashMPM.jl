abstract type AbstractExporter end
struct NoExporter <: AbstractExporter end
function write_output(exporter::NoExporter, model::MPMModel, step::Int, time::Real = model.t)
    return nothing
end

# ---------------------------------------------------------------------------- #
#                        Device-to-Host Transfer Helpers                       #
# ---------------------------------------------------------------------------- #

function _to_cpu_particles(particles)
    return StructArray{eltype(particles)}((
        id             = Array(particles.id),
        pos            = StructArray{eltype(particles.pos)}((
            x = Array(particles.pos.x),
            y = Array(particles.pos.y),
            z = Array(particles.pos.z),
        )),
        mass            = Array(particles.mass),
        V0              = Array(particles.V0),
        F               = Array(particles.F),
        mat_state       = Array(particles.mat_state),
    ))
end

function _to_cpu_grid_state(state)
    return StructArray{eltype(state)}((
        mass       = Array(state.mass),
        momentum   = StructArray{eltype(state.momentum)}((
            x = Array(state.momentum.x),
            y = Array(state.momentum.y),
            z = Array(state.momentum.z),
        )),
        wave_speed = Array(state.wave_speed),
    ))
end




# ---------------------------------------------------------------------------- #
#                            JLD2 Snapshot Exporter                            #
# ---------------------------------------------------------------------------- #

@kwdef struct JLD2Exporter <: AbstractExporter
    output_dir::String
    filename_prefix::String = "sim_"
end

function write_output(exporter::JLD2Exporter, model::MPMModel, step::Int, time::Real = model.t)
    mkpath(exporter.output_dir)
    padded_idx = Printf.@sprintf("%06d", step)
    output_file = joinpath(exporter.output_dir, "$(exporter.filename_prefix)$(padded_idx).jld2")

    cpu_model = model_to_cpu(model)
    jldsave(output_file; model = cpu_model)
    return output_file
end


# ---------------------------------------------------------------------------- #
#                                 HDF5 Exporter                                #
# ---------------------------------------------------------------------------- #
# File layout (3-vectors / 3x3-matrices are stored flat, row-major, i.e. a
# Julia array of size (3, N) / (9, N)):
#
#   particles/position, velocity, affine, F, mass, volume, id
#   particles/mat_state/...                (see extract_particle_fields.jl)
#
#   grid/node_index   (3, M)  global node indices (i, j, k), 1-based, may be <= 0
#   grid/position     (3, M)  node positions
#   grid/mass         (M,)
#   grid/momentum     (3, M)
#   grid/wave_speed   (M,)
#   grid/origin, grid/dx      grid frame (origin = position of node (1, 1, 1))
#
# The grid is a flat list of nodes, just like the particles. It is taken from the
# read state of the grid (the same state `extract_velocities` uses):
#   DenseGrid:  all nodes
#   SparseGrid: all nodes of the active tiles only (nodes outside of a BoxDomain
#               are left out). No conversion to a dense grid is needed.
# Attributes: root: time, step; particles: n_particles; grid: type, n_nodes
# (+ n_tiles, tile_size for the SparseGrid).

@kwdef struct HDF5Exporter <: AbstractExporter
    output_dir::String = "output"
    filename_prefix::String = "sim_"
    write_xdmf::Bool = true
    compression_level::Int = 3
    export_particles::Bool = true
    export_grid::Bool = false
end


# ------------------------------- Particles ---------------------------------- #
@inline function _store_matrix!(dst::AbstractMatrix, col::Int, M)
    for r in 1:3, c in 1:3          # row-major: [1,1],[1,2],[1,3],[2,1],...
        dst[3*(r-1) + c, col] = M[r, c]
    end
    return nothing
end

"""Flat particle arrays of all particle sets (in the order of the sets)."""
function _export_particle_arrays(cpu_model, ::Type{T}) where {T}
    sets  = cpu_model.particle_sets
    total = sum(ps -> length(ps.particles.mass), sets)

    position = Matrix{T}(undef, 3, total)
    velocity = Matrix{T}(undef, 3, total)
    affine   = Matrix{T}(undef, 9, total)
    F        = Matrix{T}(undef, 9, total)
    mass     = Vector{T}(undef, total)
    volume   = Vector{T}(undef, total)
    id       = Vector{Int}(undef, total)

    offset = 1
    for (set_idx, p_set) in enumerate(sets)
        N = length(p_set.particles.mass)
        N == 0 && continue
        range = offset:(offset + N - 1)

        position[1, range] .= p_set.particles.pos.x
        position[2, range] .= p_set.particles.pos.y
        position[3, range] .= p_set.particles.pos.z

        v_p, affines = extract_velocities(cpu_model.grid, p_set, cpu_model.shapefunction)
        F_p = p_set.particles.F      # Vector{SMatrix{3,3,T,9}}
        for i in 1:N
            col = offset + i - 1
            velocity[1, col] = v_p[i][1]
            velocity[2, col] = v_p[i][2]
            velocity[3, col] = v_p[i][3]
            _store_matrix!(affine, col, affines[i])
            _store_matrix!(F, col, F_p[i])
        end

        mass[range]   .= p_set.particles.mass
        volume[range] .= p_set.particles.V0
        id[range]     .= set_idx

        offset += N
    end

    return (; position, velocity, affine, F, mass, volume, id)
end


# --------------------------------- Grid ------------------------------------- #
_trim_nodes(A::AbstractMatrix, m::Int) = size(A, 2) == m ? A : A[:, 1:m]
_trim_nodes(A::AbstractVector, m::Int) = length(A) == m ? A : A[1:m]

"""
    _export_grid_nodes(grid) -> NamedTuple

Flat node list of the grid (read state) with the fields `node_index`, `position`,
`mass`, `momentum`, `wave_speed`. For a `SparseGrid` only the nodes of the active
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

    return (; node_index = ijk, position,
              mass       = vec(Array(state.mass)),
              momentum,
              wave_speed = vec(Array(state.wave_speed)))
end

function _export_grid_nodes(grid::SparseGrid)
    state = grid.state_read
    T     = eltype(state.mass)
    n     = grid.n_active_read            # only tiles 1:n_active_read are in use

    tile_keys = Array(view(grid.map_read.tile_keys, 1:n))
    order     = sortperm(tile_keys)

    mass       = Array(view(state.mass,       :, 1:n))
    wave_speed = Array(view(state.wave_speed, :, 1:n))
    px         = Array(view(state.momentum.x, :, 1:n))
    py         = Array(view(state.momentum.y, :, 1:n))
    pz         = Array(view(state.momentum.z, :, 1:n))

    M_max    = TILE_VOLUME * n
    ijk      = Matrix{Int}(undef, 3, M_max)
    position = Matrix{T}(undef, 3, M_max)
    momentum = Matrix{T}(undef, 3, M_max)
    out_mass = Vector{T}(undef, M_max)
    out_ws   = Vector{T}(undef, M_max)

    m = 0
    for t in order, l in 1:TILE_VOLUME
        i, j, k = node_index(tile_keys[t], l)
        node_in_domain(grid.domain, i, j, k) || continue    # boundary tiles may stick out of a box
        m += 1

        ijk[1, m], ijk[2, m], ijk[3, m] = i, j, k
        x = node_position(grid.origin, grid.inv_dx, i, j, k)
        position[1, m], position[2, m], position[3, m] = x[1], x[2], x[3]

        out_mass[m] = mass[l, t]
        out_ws[m]   = wave_speed[l, t]
        momentum[1, m], momentum[2, m], momentum[3, m] = px[l, t], py[l, t], pz[l, t]
    end

    return (; node_index = _trim_nodes(ijk, m),
              position   = _trim_nodes(position, m),
              mass       = _trim_nodes(out_mass, m),
              momentum   = _trim_nodes(momentum, m),
              wave_speed = _trim_nodes(out_ws, m))
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


# --------------------------------- Output ----------------------------------- #
function _write_fields!(file, group::String, fields::NamedTuple, compression_level::Int)
    for (name, arr) in pairs(fields)
        _write_dataset(file, "$(group)/$(name)", arr, compression_level)
    end
    return nothing
end

function write_output(exporter::HDF5Exporter, model::MPMModel, step::Int, time::Real = model.t)
    mkpath(exporter.output_dir)

    cpu_model = model_to_cpu(model)
    T = eltype(cpu_model.grid.origin)
    level = exporter.compression_level

    padded_idx  = Printf.@sprintf("%06d", step)
    h5_filename = "$(exporter.filename_prefix)$(padded_idx).h5"
    h5_path     = joinpath(exporter.output_dir, h5_filename)

    particles   = exporter.export_particles ? _export_particle_arrays(cpu_model, T) : nothing
    n_particles = isnothing(particles) ? 0 : size(particles.position, 2)

    # Grid nodes: no conversion to a dense grid, a SparseGrid exports its active tiles
    nodes   = exporter.export_grid ? _export_grid_nodes(cpu_model.grid) : nothing
    n_nodes = isnothing(nodes) ? 0 : size(nodes.position, 2)

    # mat_state/... fields (c, e, eos_state/*, elastic_state/*), also used for the XDMF file
    particle_fields = Dict{String, Array}()

    h5open(h5_path, "w") do file

        # 1. Particles
        if n_particles > 0
            _write_fields!(file, "particles", particles, level)

            merge!(particle_fields, _gather_particle_fields(cpu_model.particle_sets, n_particles))
            for name in sort!(collect(keys(particle_fields)))
                _write_dataset(file, "particles/" * name, particle_fields[name], level)
            end
            attributes(file["particles"])["n_particles"] = n_particles
        end

        # 2. Grid
        if n_nodes > 0
            _write_fields!(file, "grid", nodes, level)
            _write_grid_info!(file, cpu_model.grid, n_nodes)
        end

        attributes(file)["time"] = Float64(time)
        attributes(file)["step"] = step
    end

    if exporter.write_xdmf
        _write_xdmf_metadata(
            exporter.output_dir,
            exporter.filename_prefix,
            padded_idx,
            n_particles,
            n_nodes,
            T,
            Float64(time),
            particle_fields,
        )
    end

    return h5_path
end

# Generates the companion XDMF XML wrapper for ParaView ingestion.
# Particles and grid nodes are both written as point clouds (Polyvertex).
function _write_xdmf_metadata(output_dir::String, prefix::String, padded_idx::String,
                              n_particles::Int, n_nodes::Int, T::Type, time::Float64,
                              particle_fields::AbstractDict=Dict{String, Array}())

    h5_filename  = "$(prefix)$(padded_idx).h5"
    xmf_filename = "$(prefix)$(padded_idx).xmf"
    xmf_path     = joinpath(output_dir, xmf_filename)

    precision = (T == Float64) ? 8 : 4
    int_precision = sizeof(Int)

    # 1. Particles
    particles_xml = ""
    if n_particles > 0
        particles_xml = """
        <Grid Name="MPM_Particles" GridType="Uniform">
          <Topology TopologyType="Polyvertex" NumberOfElements="$(n_particles)"/>
          <Geometry GeometryType="XYZ">
            <DataItem Dimensions="$(n_particles) 3" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/particles/position
            </DataItem>
          </Geometry>
          <Attribute Name="Velocity" AttributeType="Vector" Center="Node">
            <DataItem Dimensions="$(n_particles) 3" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/particles/velocity
            </DataItem>
          </Attribute>
          <Attribute Name="Affine" AttributeType="Matrix" Center="Node">
            <DataItem Dimensions="$(n_particles) 9" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/particles/affine
            </DataItem>
          </Attribute>
          <Attribute Name="F" AttributeType="Matrix" Center="Node">
            <DataItem Dimensions="$(n_particles) 9" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/particles/F
            </DataItem>
          </Attribute>
          <Attribute Name="Mass" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(n_particles)" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/particles/mass
            </DataItem>
          </Attribute>
          <Attribute Name="Volume" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(n_particles)" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/particles/volume
            </DataItem>
          </Attribute>
          <Attribute Name="ID" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(n_particles)" NumberType="Int" Precision="$(int_precision)" Format="HDF">
              $(h5_filename):/particles/id
            </DataItem>
          </Attribute>
$(_xdmf_particle_attributes(h5_filename, particle_fields, n_particles; group="particles"))
        </Grid>
        """
    end

    # 2. Grid nodes (all nodes of a DenseGrid / the nodes of the active tiles of a SparseGrid)
    grid_xml = ""
    if n_nodes > 0
        grid_xml = """
        <Grid Name="MPM_Grid" GridType="Uniform">
          <Topology TopologyType="Polyvertex" NumberOfElements="$(n_nodes)"/>
          <Geometry GeometryType="XYZ">
            <DataItem Dimensions="$(n_nodes) 3" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/grid/position
            </DataItem>
          </Geometry>
          <Attribute Name="Mass" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(n_nodes)" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/grid/mass
            </DataItem>
          </Attribute>
          <Attribute Name="Momentum" AttributeType="Vector" Center="Node">
            <DataItem Dimensions="$(n_nodes) 3" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/grid/momentum
            </DataItem>
          </Attribute>
          <Attribute Name="WaveSpeed" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(n_nodes)" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/grid/wave_speed
            </DataItem>
          </Attribute>
        </Grid>
        """
    end

    xdmf_content = """
    <?xml version="1.0" encoding="utf-8"?>
    <Xdmf Version="3.0">
      <Domain>
        <!-- Um Partikel UND Grid gleichzeitig laden zu können, müssen sie in einer Spatial-Collection liegen -->
        <Grid Name="SimulationState" GridType="Collection" CollectionType="Spatial">
          <Time Value="$(time)"/>
$(particles_xml)
$(grid_xml)
        </Grid>
      </Domain>
    </Xdmf>
    """

    open(xmf_path, "w") do io
        write(io, strip(xdmf_content))
    end
end