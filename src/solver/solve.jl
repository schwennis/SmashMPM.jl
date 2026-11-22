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


# (unit, seconds per unit, upper limit in this unit before switching to the next one)
const _SIM_TIME_UNITS = (("ps", 1e-12, 1000.0), ("ns", 1e-9, 1000.0), ("µs", 1e-6, 1000.0),
                         ("ms", 1e-3, 1000.0), ("s", 1.0, 60.0), ("min", 60.0, 60.0),
                         ("h", 3600.0, 24.0), ("d", 86400.0, Inf))

"""
    format_sim_time(t; digits=4) -> String

Simulation time in seconds as a readable string with an adaptive unit and `digits`
significant digits, e.g. `"12.35 µs"`, `"3.5 ms"`, `"42.17 s"`, `"7.5 min"`, `"2.25 h"`, `"3.1 d"`.
"""
function format_sim_time(t::Real; digits::Int=4)
    isfinite(t) || return string(t)
    iszero(t) && return "0 s"
    a = abs(float(t))
    for (i, (unit, scale, limit)) in enumerate(_SIM_TIME_UNITS)
        v = round(a / scale; sigdigits=digits)
        # first unit in which the (rounded) value stays below its limit; tiny values stay in ps
        if v < limit || i == length(_SIM_TIME_UNITS)
            num = v >= 10.0^digits ? @sprintf("%.0f", v) : string(Float64(v))
            num = endswith(num, ".0") ? num[1:end-2] : num
            return (t < 0 ? "-" : "") * num * " " * unit
        end
    end
end


function step!(model::MPMModel)
    dt_kick = typeof(model.t)(0.5) * (dt + model.dt_previous)
    grid_kick!(model.grid, dt)

    dt = min(courant_timestep(model), model.t_max - model.t)

    # G2P2G step, reads from state read and writes to state_write
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
            println("Exported output at time $(format_sim_time(model.t)) (step $step, current dt: $(format_sim_time(dt))). eta: $(format_eta(eta_seconds))")
            flush(stdout)
        end
    end
end