abstract type AbstractExporter end


# ---------------------------------------------------------------------------- #
#                            JLD2 Snapshot Exporter                            #
# ---------------------------------------------------------------------------- #
@kwdef struct JLD2Exporter <: AbstractExporter
    output_dir::String
    filename_prefix::String = "sim_"
end

function write_output(exporter::JLD2Exporter, model::MPMModel, step::Int)
    mkpath(exporter.output_dir)
    name = "$(exporter.filename_prefix)$(Printf.@sprintf("%06d", step)).jld2"
    output_file = joinpath(exporter.output_dir, name)

    if !(model.backend isa CPU)
        model = model_to_CPU(model)
    end

    jldsave(filename=output_file, "model" => model)
end


# ---------------------------------------------------------------------------- #
#                                 VTK Exporter                                 #
# ---------------------------------------------------------------------------- #
@kwdef struct VTKExporter <: AbstractExporter
    output_dir::String
    filename_prefix::String = "sim_"
end

function write_output(exporter::VTKExporter, model::MPMModel, step::Int, time::Real)
    mkpath(exporter.output_dir)

    # Move model to CPU
    cpu_model = model_to_CPU(model)
    T = eltype(cpu_model.grid.origin)
    
    total_particles = sum(p_set -> length(p_set.particles), cpu_model.particle_sets)
    if total_particles == 0; return nothing; end

    # Arrays für den kombinierten Export allozieren (WriteVTK erwartet 3 x N Matrix für Punkte)
    all_pos = Matrix{T}(undef, 3, total_particles)
    all_vel = Matrix{T}(undef, 3, total_particles)
    all_mass = Vector{T}(undef, total_particles)
    all_vol = Vector{T}(undef, total_particles)
    all_id = Vector{Int}(undef, total_particles)
    
    offset = 1
    for (set_idx, p_set) in enumerate(cpu_model.particle_sets)
        N = length(p_set.particles)
        if N == 0
            continue
        end
        
        range = offset:(offset + N - 1)
        
        # Positionen (3 x N)
        all_pos[1, range] .= p_set.particles.pos.x
        all_pos[2, range] .= p_set.particles.pos.y
        all_pos[3, range] .= p_set.particles.pos.z
        
        # On-the-fly Geschwindigkeit interpolieren
        v_p_reconstructed = extract_velocities(cpu_model.grid, p_set, cpu_model.shapefunction)
        for i in 1:N
            all_vel[1, offset + i - 1] = v_p_reconstructed[i][1]
            all_vel[2, offset + i - 1] = v_p_reconstructed[i][2]
            all_vel[3, offset + i - 1] = v_p_reconstructed[i][3]
        end
        
        # Attribute
        all_mass[range] .= p_set.particles.mass
        all_vol[range] .= p_set.particles.initial_volume
        all_id[range] .= set_idx
        
        offset += N
    end

    # 2. VTK Topologie für Partikel definieren
    # In VTK werden lose Punktwolken als "Vertices" repräsentiert.
    # Jedes Partikel bildet eine eigene Zelle vom Typ VTKCellTypes.VTK_VERTEX.
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, [i]) for i in 1:total_particles]

    # Dateinamen generieren (z.B. "sim_data_00123.vtu")
    padded_idx = lpad(step, 5, "0")
    full_path = "$(filename_prefix)_$(padded_idx)"

    # 3. Datei im binären XML-Format mit standardmäßiger Zlib-Kompression öffnen
    vtk_grid(full_path, all_pos, cells, append=true, ascii=false) do vtk
        
        # Punktdaten (Point Data) anhängen – ParaView interpoliert diese sauber
        vtk["Velocity", VTKPointData()] = all_vel
        vtk["Mass", VTKPointData()]     = all_mass
        vtk["Volume", VTKPointData()]   = all_vol
        vtk["ID", VTKPointData()]       = all_id
        # Metadaten als globale Felddaten anhängen (z.B. die Simulationszeit)
        vtk["TimeValue", VTKFieldData()] = cpu_model.t
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
    compression_level::Int = 3 # 0 (keine) bis 9 (maximal) für gzip
end

function write_output(exporter::HDF5Exporter, model::MPMModel, step::Int, time::Real)
    mkpath(exporter.output_dir)

    # 1. Modell auf die CPU holen (analog zum VTK-Exporter)
    cpu_model = model_to_CPU(model)
    T = eltype(cpu_model.grid.origin)
    
    total_particles = sum(p_set -> length(p_set.particles), cpu_model.particle_sets)
    if total_particles == 0
        return nothing
    end

    # Arrays für den kombinierten Export allozieren
    all_pos  = Matrix{T}(undef, 3, total_particles)
    all_vel  = Matrix{T}(undef, 3, total_particles)
    all_mass = Vector{T}(undef, total_particles)
    all_vol  = Vector{T}(undef, total_particles)
    all_id   = Vector{Int}(undef, total_particles)
    
    offset = 1
    for (set_idx, p_set) in enumerate(cpu_model.particle_sets)
        N = length(p_set.particles)
        if N == 0
            continue
        end
        
        range = offset:(offset + N - 1)
        
        # Positionen (3 x N) extrahieren
        all_pos[1, range] .= p_set.particles.pos.x
        all_pos[2, range] .= p_set.particles.pos.y
        all_pos[3, range] .= p_set.particles.pos.z
        
        # On-the-fly Geschwindigkeit aus dem Gitter interpolieren (spart VRAM im Solver!)
        v_p_reconstructed = _reconstruct_velocities_cpu(p_set, cpu_model.grid, cpu_model.shapefunction)
        for i in 1:N
            all_vel[1, offset + i - 1] = v_p_reconstructed[i][1]
            all_vel[2, offset + i - 1] = v_p_reconstructed[i][2]
            all_vel[3, offset + i - 1] = v_p_reconstructed[i][3]
        end
        
        # Weitere Attribute zuweisen
        all_mass[range] .= p_set.particles.mass
        all_vol[range]  .= p_set.particles.initial_volume
        all_id[range]   .= set_idx
        
        offset += N
    end

    # Dateinamen mit sauberem Padding generieren (z.B. "sim_000123.h5")
    padded_idx = Printf.@sprintf("%06d", step)
    h5_filename = "$(exporter.filename_prefix)$(padded_idx).h5"
    h5_path = joinpath(exporter.output_dir, h5_filename)

    # 2. HDF5-Datei im Binärmodus schreiben
    h5open(h5_path, "w") do file
        if exporter.compression_level > 0
            # Mit nativer gzip-Kompression schreiben, um massiv Speicherplatz zu sparen
            # 'blosc' ist oft schneller, benötigt aber das Paket Blosc.jl. gzip ist built-in.
            file["position",  chunk=(3, min(total_particles, 1024)), compress=exporter.compression_level] = all_pos
            file["velocity",  chunk=(3, min(total_particles, 1024)), compress=exporter.compression_level] = all_vel
            file["mass",      chunk=(min(total_particles, 1024),),   compress=exporter.compression_level] = all_mass
            file["volume",    chunk=(min(total_particles, 1024),),   compress=exporter.compression_level] = all_vol
            file["id",        chunk=(min(total_particles, 1024),),   compress=exporter.compression_level] = all_id
        else
            file["position"] = all_pos
            file["velocity"] = all_vel
            file["mass"]     = all_mass
            file["volume"]   = all_vol
            file["id"]       = all_id
        end
        
        # Globale Simulations-Metadaten als HDF5-Attribute anhängen
        attributes(file)["time"]  = cpu_model.t
        attributes(file)["cycle"] = step
    end

    # 3. XDMF-Wrapper-Datei schreiben (erlaubt Plug-and-Play in ParaView)
    if exporter.write_xdmf
        _write_xdmf_metadata(exporter.output_dir, exporter.filename_prefix, padded_idx, total_particles, T)
    end

    return h5_path
end

# Interner Helper, der das XML-XDMF-File generiert
function _write_xdmf_metadata(output_dir::String, prefix::String, padded_idx::String, total_particles::Int, T::Type)
    h5_filename  = "$(prefix)$(padded_idx).h5"
    xmf_filename = "$(prefix)$(padded_idx).xmf"
    xmf_path     = joinpath(output_dir, xmf_filename)
    
    precision = (T == Float64) ? 8 : 4
    
    # WICHTIGER ARCHITEKTUR-HINWEIS FÜR JULIA (Column-Major):
    # Da ein Julia-Array der Form (3, N) im Speicher liegt, "sieht" die C-basierte HDF5/XDMF-Bibliothek
    # dies automatisch gespiegelt als eine Matrix der Dimension (N, 3). 
    # Daher definieren wir im XDMF Dimensions="$total_particles 3".
    
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