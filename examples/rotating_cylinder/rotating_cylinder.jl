# ==============================================================================
# Example: Angular Momentum Conservation in MLS-MPM
#
# This script demonstrates the setup of a rotating cylinder using the SmashMPM
# package and verifies the good conservation of angular momentum over time.
#
# Requirements: SmashMPM, StaticArrays, LinearAlgebra, CairoMakie, LaTeXStrings
# 
# ==============================================================================

using SmashMPM
using StaticArrays
using LinearAlgebra: norm, cross
using CairoMakie
using LaTeXStrings


# ---------------------------------------------------------------------------- #
#                                 Configuration                                #
# ---------------------------------------------------------------------------- #
const RADIUS = 1.0
const HEIGHT = 1.0
const DENSITY = 1000.0
const YOUNG_MODULUS = 1e5
const POISSON_RATIO = 0.3
const ROTATION_SPEED = 2π / 5.0  # radians per second

const DX = 0.1
const T_MAX = 120.0

"""
    calculate_angular_momentum(model::MPMModel)

Calculates the total angular momentum of the system based on the grid state.
"""
function calculate_angular_momentum(model::MPMModel)
    gridstate = model.grid.state_old
    L = zero(SVector{3, Float64})
    
    for i in CartesianIndices(gridstate.mass)
        mass = gridstate.mass[i]
        if mass > 0.0
            pos = model.grid.origin .- 1 .+ SVector{3, Float64}(i.I) ./ model.grid.inv_dx
            vel = gridstate.momentum[i] / mass
            L += mass * cross(pos, vel)
        end
    end

    return L
end

function main()
    # --- 1. Simulation Setup ---
    center = SVector{3, Float64}(0.0, 0.0, 0.0)
    R = RADIUS
    h = HEIGHT
    euler_angles = SVector{3, Float64}(0.0, 0.0, 0.0)
    shape1 = Cylinder(radius=R, height=h, center=center, euler_angles=euler_angles)
    
    linear_vel = SVector{3, Float64}(0.0, 0.0, 0.0)
    rot_speed = 2π / 5.0
    rotational_vec = SVector{3, Float64}(0.0, 0.0, rot_speed)
    
    mat = NeoHookean(ρ=1000.0, E=1e5, ν=0.3)
    body = Body(shape1, linear_vel, rotational_vec, mat)
    bodies = (body,)
    
    setup = SimulationSetup(dx=0.1, t_max=120.0)
    model = build_mpm_model(bodies, setup)
    
    N_particles = length(model.particle_sets[1].particles.pos)
    dims = size(model.grid.state_old.mass)
    println("Model built successfully with $N_particles particles and a grid of size $dims.")

    # --- 2. Data Logging Setup ---
    ang_moms = SVector{3, Float64}[]
    ts = Float64[]

    extraction_interval = 0.5
    last_extraction_time = 0.0

    println("Starting simulation...")
    
    # --- 3. Main Time Loop ---
    while model.t < model.t_max
        dt = SmashMPM.courant_timestep(model, 0.6)
        
        SmashMPM.g2p2g!(model, dt)
        SmashMPM.grid_reset!(model.grid)
        
        model.t += dt
        
        if model.t - last_extraction_time >= extraction_interval
            # Use \r to overwrite the line in the console for cleaner output
            print("Simulation time: $(round(model.t, digits=4)) / $(model.t_max)      \r")
            push!(ang_moms, calculate_angular_momentum(model))
            push!(ts, model.t)
            last_extraction_time = model.t
        end
    end
    println("\nSimulation completed. Total time steps to be plotted: $(length(ts))")

    # --- 4. Post-Processing & Plotting ---
    L0_norm = norm(ang_moms[1])
    rel_log_err = -log10.(abs.((norm.(ang_moms) .- L0_norm) ./ L0_norm))

    # Breite angepasst für eine einzelne Grafik (z.B. quadratischer oder 800x600)
    fig = Figure(size = (800, 600), fontsize = 18)

    # Einziger Axis-Plot für den Log-Error
    ax = Axis(fig[1, 1], 
        title = "Angular Momentum Conservation in MLS-MPM",
        xlabel = "Time [s]", 
        ylabel = L"-\log_{10}\left|\frac{|\vec{L}| - |\vec{L}_0|}{|\vec{L}_0|}\right|")

    # Plot Daten (ab dem 2. Zeitschritt, da der erste log(0) wäre)
    scatter!(ax, ts[2:end], rel_log_err[2:end], 
             color = :navyblue, marker = :xcross, markersize = 12, label = "Negative Log Error")

    # Filtert eventuelle Inf-Werte für die korrekte Skalierung der Y-Achse
    valid_errors = filter(x -> isfinite(x), rel_log_err[2:end])
    if !isempty(valid_errors)
        ylims!(ax, 0, maximum(valid_errors) * 1.2)
    end

    # Legende hinzufügen
    axislegend(ax, position = :rb)

    # Grafik speichern
    output_filename = "angular_momentum_log_error.png"
    save(output_filename, fig)
    println("Plot saved to: $output_filename")
end

# Execute the script
main()