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
#                                 HDF5 Exporter                                #
# ---------------------------------------------------------------------------- #

@kwdef struct HDF5Exporter <: AbstractExporter
    output_dir::String = "output"
    filename_prefix::String = "sim_"
    write_xdmf::Bool = true
    compression_level::Int = 3
    export_particles::Bool = true
    export_grid::Bool = false
end

function write_output(exporter::HDF5Exporter, model::MPMModel, step::Int, time::Real = model.t)
    mkpath(exporter.output_dir)
 
    cpu_model = model_to_CPU(model)
    T = eltype(cpu_model.grid.origin)
    padded_idx = Printf.@sprintf("%06d", step)
    h5_filename = "$(exporter.filename_prefix)$(padded_idx).h5"
    h5_path = joinpath(exporter.output_dir, h5_filename)
 
    total_particles = sum(p_set -> length(p_set.particles.mass), cpu_model.particle_sets)
 
    # mat_state/... fields (c, eos_state/*, strength_state/*), also used for the XDMF file
    particle_fields = Dict{String, Array}()
 
    h5open(h5_path, "w") do file
 
        # 1. Partikel-Export
        if exporter.export_particles && total_particles > 0
            all_pos  = Matrix{T}(undef, 3, total_particles)
            all_vel  = Matrix{T}(undef, 3, total_particles)
            all_affines = Matrix{T}(undef, 9, total_particles)
            all_F    = Matrix{T}(undef, 9, total_particles)
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
 
                v_p, affines = extract_velocities(cpu_model.grid, p_set, cpu_model.shapefunction)
                F_p = p_set.particles.F      # Vector{SMatrix{3,3,T,9}}
                for i in 1:N
                    col = offset + i - 1
                    all_vel[1, col] = v_p[i][1]
                    all_vel[2, col] = v_p[i][2]
                    all_vel[3, col] = v_p[i][3]
                    for r in 1:3, c in 1:3      # row-major: [1,1],[1,2],[1,3],[2,1],...
                        all_affines[3*(r-1) + c, col] = affines[i][r, c]
                        all_F[3*(r-1) + c, col]       = F_p[i][r, c]
                    end
                end
 
                all_mass[range] .= p_set.particles.mass
                all_vol[range]  .= p_set.particles.initial_volume
                all_id[range]   .= set_idx
 
                offset += N
            end
 
            chunk_size = min(total_particles, 1024)
            if exporter.compression_level > 0
                file["position",  chunk=(3, chunk_size), compress=exporter.compression_level] = all_pos
                file["velocity",  chunk=(3, chunk_size), compress=exporter.compression_level] = all_vel
                file["affine",    chunk=(9, chunk_size), compress=exporter.compression_level] = all_affines
                file["F",         chunk=(9, chunk_size), compress=exporter.compression_level] = all_F
                file["mass",      chunk=(chunk_size,),   compress=exporter.compression_level] = all_mass
                file["volume",    chunk=(chunk_size,),   compress=exporter.compression_level] = all_vol
                file["id",        chunk=(chunk_size,),   compress=exporter.compression_level] = all_id
            else
                file["position"] = all_pos
                file["velocity"] = all_vel
                file["affine"]   = all_affines
                file["F"]        = all_F
                file["mass"]     = all_mass
                file["volume"]   = all_vol
                file["id"]       = all_id
            end
 
            # Material state: c, eos_state/*, strength_state/*
            merge!(particle_fields, _gather_particle_fields(cpu_model.particle_sets, total_particles))
            for name in sort!(collect(keys(particle_fields)))
                _write_dataset(file, name, particle_fields[name], exporter.compression_level)
            end
        end
 
        # 2. Grid-Export
        if exporter.export_grid
            grid_state = _to_cpu_grid_state(cpu_model.grid.state_old)
            
            # Grid-Parameter ablegen
            file["grid/origin"] = [cpu_model.grid.origin[1], cpu_model.grid.origin[2], cpu_model.grid.origin[3]]
            file["grid/dx"]     = 1.0 / cpu_model.grid.inv_dx
            
            # Momentum-Komponenten zu einem 4D-Array (Vektorfeld) zusammenfassen
            Nx, Ny, Nz = size(grid_state.mass)
            momentum_vec = Array{T, 4}(undef, 3, Nx, Ny, Nz)
            momentum_vec[1, :, :, :] .= grid_state.momentum.x
            momentum_vec[2, :, :, :] .= grid_state.momentum.y
            momentum_vec[3, :, :, :] .= grid_state.momentum.z
            
            # Grid-Felder speichern
            if exporter.compression_level > 0
                file["grid/mass", compress=exporter.compression_level] = grid_state.mass
                file["grid/momentum", compress=exporter.compression_level] = momentum_vec
            else
                file["grid/mass"]       = grid_state.mass
                file["grid/momentum"]   = momentum_vec
            end
        end
 
        attributes(file)["time"]  = Float64(time)
        attributes(file)["step"] = step
    end
 
    if exporter.write_xdmf
        grid_dims = (0, 0, 0)
        origin = (0.0, 0.0, 0.0)
        dx = 1.0
        
        if exporter.export_grid
            grid_dims = size(cpu_model.grid.state_old.mass)
            origin = cpu_model.grid.origin
            dx = 1.0 / cpu_model.grid.inv_dx
        end
        
        _write_xdmf_metadata(
            exporter.output_dir, 
            exporter.filename_prefix, 
            padded_idx, 
            total_particles, 
            T,
            Float64(time),
            exporter.export_particles,
            exporter.export_grid,
            grid_dims,
            origin,
            dx,
            particle_fields
        )
    end
 
    return h5_path
end
 
# Generates the companion XDMF XML wrapper for ParaView ingestion
function _write_xdmf_metadata(output_dir::String, prefix::String, padded_idx::String, 
                              total_particles::Int, T::Type, time::Float64,
                              export_particles::Bool=true, export_grid::Bool=false, 
                              grid_dims=(0,0,0), origin=(0.0,0.0,0.0), dx=1.0,
                              particle_fields::AbstractDict=Dict{String, Array}())
 
    h5_filename  = "$(prefix)$(padded_idx).h5"
    xmf_filename = "$(prefix)$(padded_idx).xmf"
    xmf_path     = joinpath(output_dir, xmf_filename)
 
    precision = (T == Float64) ? 8 : 4
    int_precision = sizeof(Int) 
 
    # 1. Partikel-Block (Polyvertex)
    particles_xml = ""
    if export_particles && total_particles > 0
        particles_xml = """
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
          <Attribute Name="Affine" AttributeType="Matrix" Center="Node">
            <DataItem Dimensions="$(total_particles) 9" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/affine
            </DataItem>
          </Attribute>
          <Attribute Name="F" AttributeType="Matrix" Center="Node">
            <DataItem Dimensions="$(total_particles) 9" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/F
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
            <DataItem Dimensions="$(total_particles)" NumberType="Int" Precision="$(int_precision)" Format="HDF">
              $(h5_filename):/id
            </DataItem>
          </Attribute>
$(_xdmf_particle_attributes(h5_filename, particle_fields, total_particles))
        </Grid>
        """
    end
 
    # 2. Gitter-Block (3D Uniform Grid)
    grid_xml = ""
    if export_grid
        Nz, Ny, Nx = grid_dims[3], grid_dims[2], grid_dims[1]
        Oz, Oy, Ox = origin[3], origin[2], origin[1]
        
        grid_xml = """
        <Grid Name="MPM_Grid" GridType="Uniform">
          <Topology TopologyType="3DCoRectMesh" Dimensions="$(Nz) $(Ny) $(Nx)"/>
          <Geometry GeometryType="ORIGIN_DXDYDZ">
            <!-- Origin: Z Y X -->
            <DataItem Dimensions="3" NumberType="Float" Precision="$(precision)" Format="XML">
              $(Oz) $(Oy) $(Ox)
            </DataItem>
            <!-- Spacing: dZ dY dX -->
            <DataItem Dimensions="3" NumberType="Float" Precision="$(precision)" Format="XML">
              $(dx) $(dx) $(dx)
            </DataItem>
          </Geometry>
          <Attribute Name="Mass" AttributeType="Scalar" Center="Node">
            <DataItem Dimensions="$(Nz) $(Ny) $(Nx)" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/grid/mass
            </DataItem>
          </Attribute>
          <Attribute Name="Momentum" AttributeType="Vector" Center="Node">
            <DataItem Dimensions="$(Nz) $(Ny) $(Nx) 3" NumberType="Float" Precision="$(precision)" Format="HDF">
              $(h5_filename):/grid/momentum
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
