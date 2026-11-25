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
        wave_speed      = Array(particles.wave_speed),
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
        force      = StructArray{eltype(state.force)}((
            x = Array(state.force.x),
            y = Array(state.force.y),
            z = Array(state.force.z),
        )),
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
