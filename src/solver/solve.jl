function step!(model::MPMModel)
    dt = courant_timestep(model)

    # G2P2G step, reads from old grid and writes to new grid
    g2p2g!(model, dt)
    
    # Apply external forces to the grid
    apply_external_forces!(model.external_force, model.grid, dt)

    # Apply boundary conditions to the new grid
    apply_boundary_condition!(model.boundary_condition, model.grid)
    
    # Reset grid for next step, swaps old <-> new
    grid_reset!(model.grid)

    # Update simulation time
    model.t += dt

    return dt
end



function solve!(model::MPMModel, setup::SimulationSetup)
    T = eltype(model.grid.mass)
    
    time_since_last_export = T(0.0)
    step = 0
    while model.t < model.t_max
        dt = step!(model)

        time_since_last_export += dt
        step += 1

        if !(setup.exporter isa NoExporter) && time_since_last_export >= model.export_time_interval
            time_since_last_export = T(0.0)
            write_output(setup.exporter, model, step, model.t)
            model.next_export_time += setup.export_time_interval
            println("Exported output at time %.4f (step %d)", model.t, step)
        end
    end
end