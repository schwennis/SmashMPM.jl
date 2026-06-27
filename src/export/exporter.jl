abstract type AbstractExporter end


# ---------------------------------------------------------------------------- #
#                            JLD2 Snapshot Exporter                            #
# ---------------------------------------------------------------------------- #
@kwdef struct JLD2Exporter <: AbstractExporter
    output_dir::String
    filename_prefix::String = "sim_"
end

function write_output(exporter::JLD2Exporter, model::Model, step::Int)
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
        v_p_reconstructed = _reconstruct_velocities_cpu(p_set, cpu_model.grid, cpu_model.shapefunction)
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
    padded_idx = lpad(frame_idx, 5, "0")
    full_path = "$(base_filename)_$(padded_idx)"

    # 3. Datei im binären XML-Format mit standardmäßiger Zlib-Kompression öffnen
    vtk_grid(full_path, all_pos, cells, append=true, ascii=false) do vtk
        
        # Punktdaten (Point Data) anhängen – ParaView interpoliert diese sauber
        vtk["Velocity", VTKPointData()] = all_vel
        vtk["Mass", VTKPointData()]     = all_mass
        vtk["Volume", VTKPointData()]   = all_vol
        vtk["ID", VTKPointData()]       = all_id
        # Metadaten als globale Felddaten anhängen (z.B. die Simulationszeit)
        vtk["TimeValue", VTKFieldData()] = cpu_model.t
        vtk["Cycle", VTKFieldData()]     = frame_idx
    end

    return "$(full_path).vtu"
end