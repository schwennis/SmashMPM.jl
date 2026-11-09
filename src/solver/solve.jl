using Printf

"""
    format_eta(seconds) -> String

Formats duration into a string containing dd:hh:mm or hh:mm depending on the duration.
"""
function format_eta(seconds::Real)
    (isfinite(seconds) && seconds >= 0) || return "--:--"
    total_minutes = round(Int, seconds / 60)
    days, rem_minutes = divrem(total_minutes, 24 * 60)
    hours, minutes = divrem(rem_minutes, 60)
    return days > 0 ? @sprintf("%d:%02d:%02d", days, hours, minutes) :
                      @sprintf("%02d:%02d", hours, minutes)
end


function step!(model::MPMModel)
    dt = min(courant_timestep(model), model.t_max - model.t)

    # G2P2G step, reads from old grid and writes to new grid
    g2p2g!(model, dt)
    
    # Apply external forces to the grid
    apply_external_forces!(model.external_force, model.grid, dt)

    # Apply boundary conditions to the new grid
    apply_boundary_condition!(model.boundary_condition, model.grid)
    
    # Reset grid for next step, swaps old <-> new
    # (for SparseGrid: also activates the tiles for the next g2p2g from the new particle positions)
    grid_reset!(model.grid, model.particle_sets, model.shapefunction)

    # Update simulation time
    model.t += dt

    return dt
end



function solve!(model::MPMModel, setup::SimulationSetup)
    T = eltype(model.grid.state_read.mass)

    t_start = model.t
    
    time_since_last_export = T(0.0)
    step = 0
    
    println("Exported output at time $(round(model.t, digits=4)) (step $step).     ")
    write_output(setup.exporter, model, step, model.t)
    
    real_time_start = time()
    while model.t < model.t_max
        dt = step!(model)

        time_since_last_export += dt
        step += 1

        if !(setup.exporter isa NoExporter) && (time_since_last_export >= setup.export_time_interval || model.t ≈ model.t_max)
            elapsed_real_time = time() - real_time_start
            eta_seconds = elapsed_real_time / (model.t - t_start) * (model.t_max - model.t)

            time_since_last_export = T(0.0)
            write_output(setup.exporter, model, step, model.t)
            println("Exported output at time $(round(model.t, digits=4)) (step $step, current dt: $(@sprintf("%.2e", dt))). eta: $(format_eta(eta_seconds))")
            flush(stdout)
        end
    end
end