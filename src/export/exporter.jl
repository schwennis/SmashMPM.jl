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
    output_dir::String
    filename_prefix::String = "sim_"
    write_xdmf::Bool = true
    compression_level::Int = 3
    export_particles::Bool = true
    export_grid::Bool = false
end
