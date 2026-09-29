abstract type AbstractExporter end

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
        mass           = Array(particles.mass),
        initial_volume = Array(particles.initial_volume),
        F              = Array(particles.F),
        mat_state      = Array(particles.mat_state),
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

# Interpolate nodal velocities to particles on the CPU
function extract_velocities(grid::DenseGrid, particle_set::SoAParticleSet, spline::AbstractShapeFunction)
    grid_state = grid.state_old
    T = eltype(grid_state.mass)

    num_particles = length(particle_set.particles.mass)
    velocities = Vector{SVector{3, T}}(undef, num_particles)

    inv_dx = grid.inv_dx
    origin = grid.origin

    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    @inbounds for p_idx in 1:num_particles
        pos = particle_set.particles.pos[p_idx]
        vel = zero(SVector{3, T})

        grid_pos = get_grid_position(pos, inv_dx, origin)
        base_node = get_support_base(spline, grid_pos)

        for di in iterator_i, dj in iterator_j, dk in iterator_k
            i = base_node[1] + di
            j = base_node[2] + dj
            k = base_node[3] + dk

            if !checkbounds(Bool, grid_state.mass, i, j, k)
                continue
            end

            natural_coords = grid_pos - SVector(i, j, k)
            N = shapefunction(spline, natural_coords)

            m_node = grid_state.mass[i, j, k]
            if m_node > eps(T) * 100
                v_grid = grid_state.momentum[i, j, k] / m_node
                vel += N * v_grid
            end
        end
        velocities[p_idx] = vel
    end

    return velocities
end

const _reconstruct_velocities_cpu = extract_velocities


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

    cpu_model = model_to_CPU(model)
    jldsave(output_file; model = cpu_model)
    return output_file
end


# ---------------------------------------------------------------------------- #
#                                 VTK Exporter                                 #
# ---------------------------------------------------------------------------- #

@kwdef struct VTKExporter <: AbstractExporter
    output_dir::String
    filename_prefix::String = "sim_"
end

function write_output(exporter::VTKExporter, model::MPMModel, step::Int, time::Real = model.t)
    mkpath(exporter.output_dir)

    cpu_model = model_to_CPU(model)
    T = eltype(cpu_model.grid.origin)

    total_particles = sum(p_set -> length(p_set.particles.mass), cpu_model.particle_sets)
    if total_particles == 0
        return nothing
    end

    # Preallocate contiguous flat arrays for WriteVTK (3 x N for point coords)
    all_pos  = Matrix{T}(undef, 3, total_particles)
    all_vel  = Matrix{T}(undef, 3, total_particles)
    all_mass = Vector{T}(undef, total_particles)
    all_vol  = Vector{T}(undef, total_particles)
    all_id   = Vector{Int}(undef, total_particles)

    offset = 1
    for (set_idx, p_set) in enumerate(cpu_model.particle_sets)
        N = length(p_set.particles.mass)
        N == 0 && continue

        range = offset:(offset + N - 1)

        all_pos[1, range] .= p_set.particles.pos.x
        all_pos[2, range] .= p_set.particles.pos.y
        all_pos[3, range] .= p_set.particles.pos.z

        v_p = extract_velocities(cpu_model.grid, p_set, cpu_model.shapefunction)
        for i in 1:N
            all_vel[1, offset + i - 1] = v_p[i][1]
            all_vel[2, offset + i - 1] = v_p[i][2]
            all_vel[3, offset + i - 1] = v_p[i][3]
        end

        all_mass[range] .= p_set.particles.mass
        all_vol[range]  .= p_set.particles.initial_volume
        all_id[range]   .= set_idx

        offset += N
    end

    # 1-tuple avoids vector allocations inside MeshCell
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, (i,)) for i in 1:total_particles]

    padded_idx = Printf.@sprintf("%06d", step)
    full_path = joinpath(exporter.output_dir, "$(exporter.filename_prefix)$(padded_idx)")

    vtk_grid(full_path, all_pos, cells, append=true, ascii=false) do vtk
        vtk["Velocity", VTKPointData()] = all_vel
        vtk["Mass", VTKPointData()]     = all_mass
        vtk["Volume", VTKPointData()]   = all_vol
        vtk["ID", VTKPointData()]       = all_id

        vtk["TimeValue", VTKFieldData()] = T(time)
        vtk["Cycle", VTKFieldData()]     = step
    end

    return "$(full_path).vtu"
end


# ---------------------------------------------------------------------------- #
#                                 HDF5 Exporter                                #
# ---------------------------------------------------------------------------- #

@kwdef struct HDF5Exporter <: AbstractExporter
    output_dir::String
    filename_prefix::String = "sim_"
    write_xdmf::Bool = true
    compression_level::Int = 3
end

function write_output(exporter::HDF5Exporter, model::MPMModel, step::Int, time::Real = model.t)
    mkpath(exporter.output_dir)

    cpu_model = model_to_CPU(model)
    T = eltype(cpu_model.grid.origin)

    total_particles = sum(p_set -> length(p_set.particles.mass), cpu_model.particle_sets)
    if total_particles == 0
        return nothing
    end

    all_pos  = Matrix{T}(undef, 3, total_particles)
    all_vel  = Matrix{T}(undef, 3, total_particles)
    all_mass = Vector{T}(undef, total_particles)
    all_vol  = Vector{T}(undef, total_particles)
    all_id   = Vector{Int}(undef, total_particles)

    offset = 1
    for (set_idx, p_set) in enumerate(cpu_model.particle_sets)
        N = length(p_set.particles.mass)
        N == 0 && continue

        range = offset:(offset + N - 1)

        all_pos[1, range] .= p_set.particles.pos.x
        all_pos[2, range] .= p_set.particles.pos.y
        all_pos[3, range] .= p_set.particles.pos.z

        v_p = extract_velocities(cpu_model.grid, p_set, cpu_model.shapefunction)
        for i in 1:N
            all_vel[1, offset + i - 1] = v_p[i][1]
            all_vel[2, offset + i - 1] = v_p[i][2]
            all_vel[3, offset + i - 1] = v_p[i][3]
        end

        all_mass[range] .= p_set.particles.mass
        all_vol[range]  .= p_set.particles.initial_volume
        all_id[range]   .= set_idx

        offset += N
    end

    padded_idx = Printf.@sprintf("%06d", step)
    h5_filename = "$(exporter.filename_prefix)$(padded_idx).h5"
    h5_path = joinpath(exporter.output_dir, h5_filename)

    h5open(h5_path, "w") do file
        chunk_size = min(total_particles, 1024)
        if exporter.compression_level > 0
            file["position",  chunk=(3, chunk_size), compress=exporter.compression_level] = all_pos
            file["velocity",  chunk=(3, chunk_size), compress=exporter.compression_level] = all_vel
            file["mass",      chunk=(chunk_size,),   compress=exporter.compression_level] = all_mass
            file["volume",    chunk=(chunk_size,),   compress=exporter.compression_level] = all_vol
            file["id",        chunk=(chunk_size,),   compress=exporter.compression_level] = all_id
        else
            file["position"] = all_pos
            file["velocity"] = all_vel
            file["mass"]     = all_mass
            file["volume"]   = all_vol
            file["id"]       = all_id
        end

        attributes(file)["time"]  = Float64(time)
        attributes(file)["cycle"] = step
    end

    if exporter.write_xdmf
        _write_xdmf_metadata(exporter.output_dir, exporter.filename_prefix, padded_idx, total_particles, T)
    end

    return h5_path
end

# Generates the companion XDMF XML wrapper for ParaView ingestion
function _write_xdmf_metadata(output_dir::String, prefix::String, padded_idx::String, total_particles::Int, T::Type)
    h5_filename  = "$(prefix)$(padded_idx).h5"
    xmf_filename = "$(prefix)$(padded_idx).xmf"
    xmf_path     = joinpath(output_dir, xmf_filename)

    precision = (T == Float64) ? 8 : 4

    # Julia (3, N) column-major layout maps to (N, 3) in row-major C/HDF5 readers
    xdmf_content = """
    <?xml version="1.0" ?>
    <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
    <Xdmf Version="3.0">
      <Domain>
        <Grid Name="MPM_Particles" GridType="Uniform">
          <Topology TopologyType="Polyvertex" NumberOfElements="$(total_particles)"/>
          <Geometry GeometryType="XYZ">
            <DataItem Dimensions="$(total_particles) 3" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/position
            </DataItem>
          </Geometry>
          <Attribute Name="Velocity" AttributeType="Vector" Center="Node">
            <DataItem Dimensions="$(total_particles) 3" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/velocity
            </DataItem>
          </Attribute>
          <Attribute Name="Mass" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(total_particles)" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/mass
            </DataItem>
          </Attribute>
          <Attribute Name="Volume" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(total_particles)" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/volume
            </DataItem>
          </Attribute>
          <Attribute Name="ID" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(total_particles)" NumberType="Int" Precision="8" Format="HDF">
              $(h5_filename):/id
            </DataItem>
          </Attribute>
        </Grid>
      </Domain>
    </Xdmf>
    """

    open(xmf_path, "w") do io
        write(io, strip(xdmf_content))
    end
end