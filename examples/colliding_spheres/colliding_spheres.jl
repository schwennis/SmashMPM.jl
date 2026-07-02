# =============================================================================
# Sphere Impact Example — SmashMPM.jl
#
# Simulates a small, stiff projectile sphere striking a larger, softer target
# sphere (both modeled as NeoHookean elastic solids) with an off-center
# impact parameter, and renders two synchronized videos of the simulation:
#
#   1. mpm_impact_perspective.mp4 — full 3D particle view       (GLMakie)
#   2. mpm_impact_xz_slice.mp4    — 2D XZ slice through the domain, with a
#                                    grid-mass heatmap and nodal velocity
#                                    field overlay                (CairoMakie)
#
# Requirements: SmashMPM, StaticArrays, GLMakie, CairoMakie, FFMPEG
# Usage:        julia --project sphere_impact_example.jl
# =============================================================================
 
using SmashMPM
using StaticArrays
using GLMakie
import CairoMakie   # `import`, not `using` — avoids name clashes with GLMakie
using FFMPEG
 
# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
const FPS                 = 30       # output video framerate
const TIME_FACTOR         = 0.2      # slow-motion factor applied to sim time
const MAX_FRAMES          = 2000     # safety cap on the number of rendered frames
const MAX_PARTICLES_WARN  = 100_000  # print a performance warning above this count
const MAX_PARTICLES_ERROR = 200_000  # abort above this count
 
const OUTPUT_PERSPECTIVE = "mpm_impact_perspective.mp4"
const OUTPUT_SLICE       = "mpm_impact_xz_slice.mp4"
 
# -----------------------------------------------------------------------
# Problem setup
# -----------------------------------------------------------------------
 
"""
    build_bodies()
 
Construct the target and projectile bodies for this example: a large, soft
NeoHookean target sphere at rest, and a smaller, stiffer projectile sphere
fired at it with an off-center impact parameter. Returns the two bodies
along with the initial separation distance and impact speed (needed later
to size the simulation duration and the velocity color scale).
"""
function build_bodies()
    # Target
    R_target = 0.5
    center_target = SVector{3,Float64}(0.0, 0.0, 0.0)
    mat_target = NeoHookean(ρ=1000.0, E=5000, ν=0.47)
    shape_target = SmashMPM.Sphere(center=center_target, radius=R_target)
    body_target = Body(shape_target, zero(center_target), zero(center_target), mat_target)
 
    # Projectile
    R_projectile = 0.1
    dist = R_target + R_projectile + 0.1
    impact_parameter = 0.5 * R_target
    mat_projectile = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
    vel_0 = 10.0
    v_projectile = vel_0 * SVector(-1.0, 0.0, 0.0)
    center_projectile = center_target + SVector(dist, 0.0, impact_parameter)
    shape_projectile = SmashMPM.Sphere(center=center_projectile, radius=R_projectile)
    body_projectile = Body(shape_projectile, v_projectile, zero(v_projectile), mat_projectile)
 
    return body_target, body_projectile, dist, vel_0
end
 

"""
    validate_model(model, dx)
 
Run a handful of sanity checks on a freshly built MPM model (identity
deformation gradients, positive mass/volume, correct initial particle
volume, and consistent density) and return the total particle count.
Aborts if the particle count exceeds the supported maximum.
"""
function validate_model(model, dx)
    particle_sets = model.particle_sets
 
    @assert particle_sets[1].particles.F[1] == one(SMatrix{3,3,Float64,9}) "Initial deformation gradient for target is not identity"
    @assert particle_sets[2].particles.F[1] == one(SMatrix{3,3,Float64,9}) "Initial deformation gradient for projectile is not identity"
    @assert all(particle_sets[1].particles.mass .> 0.0) "Mass for target particles is not positive"
    @assert all(particle_sets[2].particles.mass .> 0.0) "Mass for projectile particles is not positive"
    @assert all(particle_sets[1].particles.initial_volume .> 0.0) "Volume for target particles is not positive"
    @assert all(particle_sets[2].particles.initial_volume .> 0.0) "Volume for projectile particles is not positive"
    @assert particle_sets[1].particles.initial_volume[1] ≈ (dx / 2)^3 "Initial volume for target particles is not correct"
    @assert particle_sets[2].particles.initial_volume[1] ≈ (dx / 2)^3 "Initial volume for projectile particles is not correct"
    @assert particle_sets[1].particles.mass[1] / particle_sets[1].particles.initial_volume[1] ≈ particle_sets[1].material.ρ "Volume and mass for target particles do not match"
 
    N_particles = length(particle_sets[1].particles.pos) + length(particle_sets[2].particles.pos)
    if N_particles > MAX_PARTICLES_WARN
        println("Warning: particle count ($N_particles) is high and may impact performance.")
    end
    if N_particles > MAX_PARTICLES_ERROR
        error("Particle count ($N_particles) exceeds the supported maximum of $MAX_PARTICLES_ERROR. Reduce resolution or increase available memory.")
    end
 
    return N_particles
end

# -----------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------
 
"""
    update_visuals!(buf_3d, buf_slice, col_slice, model, y_center, thickness,
                     target_color, projectile_color)
 
Refresh the 3D particle buffer and the (variable-length) slice buffers from
the current particle state. A particle is added to the slice buffer only if
it lies within `thickness` of the `y_center` plane.
"""
function update_visuals!(buf_3d, buf_slice, col_slice, model, y_center, thickness,
                          target_color, projectile_color)
    idx = 1
    empty!(buf_slice)
    empty!(col_slice)
 
    for (set_idx, set) in enumerate(model.particle_sets)
        c = set_idx == 1 ? target_color : projectile_color
 
        px = set.particles.pos.x
        py = set.particles.pos.y
        pz = set.particles.pos.z
 
        @inbounds for i in eachindex(px)
            buf_3d[idx] = Point3f(px[i], py[i], pz[i])
            idx += 1
 
            if abs(py[i] - y_center) <= thickness
                push!(buf_slice, CairoMakie.Point2f(px[i], pz[i]))
                push!(col_slice, c)
            end
        end
    end
end
 
"""
    compute_arrow_data(model, jy, stride, threshold, dx_grid)
 
Compute arrow positions, directions, and speeds for the nodal velocity
field on grid layer `jy`, skipping nodes whose mass is below `threshold`
and sampling every `stride`-th node in each direction.
"""
function compute_arrow_data(model, jy, stride, threshold, dx_grid)
    mass = model.grid.state_old.mass
    momx = model.grid.state_old.momentum.x
    momz = model.grid.state_old.momentum.z
    Nx, _, Nz = size(mass)
 
    ps = CairoMakie.Point2f[]
    ds = CairoMakie.Vec2f[]
    speeds = Float64[]
 
    for i in 1:stride:Nx, k in 1:stride:Nz
        m = mass[i, jy, k]
        if m > threshold
            vx = momx[i, jy, k] / m
            vz = momz[i, jy, k] / m
            spd = sqrt(vx^2 + vz^2)
            if spd > 0
                push!(ps, CairoMakie.Point2f(model.grid.origin[1] + (i - 1) * dx_grid,
                                              model.grid.origin[3] + (k - 1) * dx_grid))
                push!(ds, CairoMakie.Vec2f(vx, vz))
                push!(speeds, spd)
            end
        end
    end
 
    return ps, ds, speeds
end
 
"""
    setup_visualization(model, N_particles, vel_0, dx)
 
Build the two figures used to visualize the simulation:
 
  * `fig_persp` — a 3D perspective scatter plot of all particles (GLMakie)
  * `fig_slice` — a 2D XZ slice through the domain (CairoMakie), showing a
    grid-mass heatmap, the underlying grid lines, nodal velocity arrows,
    and the particles that fall inside the slice.
 
Returns a `NamedTuple` bundling the figures, observables, and helper data
needed to update and record frames during the simulation loop.
"""
function setup_visualization(model, N_particles, vel_0, dx)
    GLMakie.activate!() # GLMakie is the active backend for the figures created below
 
    # --- Buffers ---------------------------------------------------------
    position_buffer_3d = Vector{Point3f}(undef, N_particles)
 
    # The slice buffers are dynamically sized since the number of particles
    # crossing the slice plane varies over time.
    position_buffer_slice = CairoMakie.Point2f[]
    color_buffer_slice = CairoMakie.RGBAf[]
    sizehint!(position_buffer_slice, N_particles ÷ 5) # rough pre-allocation estimate
    sizehint!(color_buffer_slice, N_particles ÷ 5)
 
    # Particles within y = 0.0 ± slice_thickness are considered part of the slice.
    slice_y_center = 0.0
    slice_thickness = dx * 1.5
 
    target_color_cairo = CairoMakie.RGBAf(0.3, 0.9, 0.2, 0.25)
    projectile_color_cairo = CairoMakie.RGBAf(0.9, 0.2, 0.2, 0.25)
    particle_colors_3d = vcat(
        fill(RGBAf(0.3, 0.9, 0.2, 0.25), length(model.particle_sets[1].particles.pos)),
        fill(RGBAf(0.9, 0.2, 0.2, 0.25), length(model.particle_sets[2].particles.pos)),
    )
 
    update_visuals!(position_buffer_3d, position_buffer_slice, color_buffer_slice,
                     model, slice_y_center, slice_thickness,
                     target_color_cairo, projectile_color_cairo)
 
    points_obs_3d    = Observable(position_buffer_3d)
    points_obs_slice = CairoMakie.Observable(position_buffer_slice)
    colors_obs_slice = CairoMakie.Observable(color_buffer_slice)
 
    # Fixed bounding box so the axes don't rescale during the simulation.
    bounding_box = (
        model.grid.origin,
        model.grid.origin .+ size(model.grid.state_old.mass) ./ model.grid.inv_dx,
    )
 
    # --- Figure 1: 3D perspective view -----------------------------------
    fig_persp = Figure(size=(800, 600))
    ax_persp = Axis3(fig_persp[1, 1], title="MPM Impact - 3D Perspective",
                      aspect=:data, azimuth=pi / 4, elevation=pi / 8)
    meshscatter!(ax_persp, points_obs_3d, markersize=0.027, color=particle_colors_3d)
    xlims!(ax_persp, bounding_box[1][1], bounding_box[2][1])
    ylims!(ax_persp, bounding_box[1][2], bounding_box[2][2])
    zlims!(ax_persp, bounding_box[1][3], bounding_box[2][3])
    display(fig_persp)
 
    # --- Figure 2: 2D XZ slice with heatmap + velocity field -------------
    CairoMakie.activate!()
    fig_slice = CairoMakie.Figure(size=(800, 600))
    ax_slice = CairoMakie.Axis(fig_slice[1, 1], title="MPM Impact - 2D XZ Slice",
                                aspect=CairoMakie.DataAspect())
 
    # Grid node coordinates (field values live on the nodes, not cell centers).
    dx_grid = 1.0 / model.grid.inv_dx
    Nx_grid, Ny_grid, Nz_grid = size(model.grid.state_old.mass)
    xs_nodes = model.grid.origin[1] .+ (0:Nx_grid-1) .* dx_grid
    zs_nodes = model.grid.origin[3] .+ (0:Nz_grid-1) .* dx_grid
 
    # Node layer along y closest to the slice plane.
    jy = clamp(round(Int, (slice_y_center - model.grid.origin[2]) / dx_grid) + 1, 1, Ny_grid)
 
    # Grid-mass heatmap (background), centered on the grid nodes.
    mass_slice_obs = CairoMakie.Observable(Matrix{Float64}(model.grid.state_old.mass[:, jy, :]))
    col_range = (0.5 * 1000, 1.5 * 1000) .* dx^3  # arbitrary starting range; adjust to taste
    hm = CairoMakie.heatmap!(ax_slice, xs_nodes, zs_nodes, mass_slice_obs;
                              colormap=:inferno, colorrange=col_range)
    CairoMakie.Colorbar(fig_slice[1, 2], hm, label="Grid mass")
 
    # Grid overlay: node lines drawn semi-transparent on top of the heatmap so
    # the underlying grid structure stays visible.
    CairoMakie.hlines!(ax_slice, zs_nodes; color=CairoMakie.RGBAf(1, 1, 1, 0.35), linewidth=0.75)
    CairoMakie.vlines!(ax_slice, xs_nodes; color=CairoMakie.RGBAf(1, 1, 1, 0.35), linewidth=0.75)
 
    # Velocity field: arrows colored by speed, drawn with a uniform length for readability.
    arrow_stride = 2      # draw every n-th node (1 = all) for readability
    mass_threshold = 1e-8 # hide nodes with negligible mass
 
    arrow_points_init, arrow_dirs_init, arrow_speeds_init =
        compute_arrow_data(model, jy, arrow_stride, mass_threshold, dx_grid)
 
    arrow_points_obs = CairoMakie.Observable(arrow_points_init)
    arrow_dirs_obs   = CairoMakie.Observable(arrow_dirs_init)
    arrow_speeds_obs = CairoMakie.Observable(arrow_speeds_init)
 
    CairoMakie.arrows2d!(ax_slice, arrow_points_obs, arrow_dirs_obs;
        color=arrow_speeds_obs,
        colormap=:viridis,
        colorrange=(0.0, vel_0),   # fixed scale, based on the impact velocity
        lengthscale=dx_grid * 2,   # fixed, uniform arrow length
    )
    CairoMakie.Colorbar(fig_slice[1, 3], colormap=:viridis, colorrange=(0.0, vel_0),
                         label="Velocity magnitude")
 
    # Particles are drawn last, as an overlay on top of everything else.
    CairoMakie.scatter!(ax_slice, points_obs_slice, markersize=6, color=colors_obs_slice)
    CairoMakie.xlims!(ax_slice, bounding_box[1][1], bounding_box[2][1])
    CairoMakie.ylims!(ax_slice, bounding_box[1][3], bounding_box[2][3])
 
    return (;
        fig_persp, fig_slice,
        points_obs_3d, points_obs_slice, colors_obs_slice,
        position_buffer_3d, position_buffer_slice, color_buffer_slice,
        mass_slice_obs, arrow_points_obs, arrow_dirs_obs, arrow_speeds_obs,
        slice_y_center, slice_thickness,
        target_color_cairo, projectile_color_cairo,
        jy, dx_grid, arrow_stride, mass_threshold,
    )
end
 
# -----------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------
 
"""
    run_simulation!(model, vis; max_frames=MAX_FRAMES)
 
Advance the MPM simulation to `model.t_max`, recording synchronized frames
into two video streams (3D perspective and 2D slice) at a fixed framerate.
Returns the two `VideoStream`s so the caller can save them to disk.
 
Note: both Makie backends render through whichever backend is currently
active, so each backend must be re-activated immediately before its
`recordframe!` call.
"""
function run_simulation!(model, vis; max_frames=MAX_FRAMES)
    plot_interval = 1.0 / FPS * TIME_FACTOR
    last_plot_t = 0.0
 
    GLMakie.activate!()
    screen = GLMakie.display(vis.fig_persp)
    
    stream_persp = GLMakie.VideoStream(vis.fig_persp, framerate=FPS)
 
    CairoMakie.activate!()
    stream_slice = CairoMakie.VideoStream(vis.fig_slice, framerate=FPS)
 
    println("\nStarting simulation and video recording...")
 
    for frame in 1:max_frames
        print("Frame $frame, Simulation Time: $(round(model.t, digits=4))s/$(round(model.t_max, digits=4))s      \r")
 
        try
            while model.t < model.t_max && (model.t - last_plot_t < plot_interval)
                dt = SmashMPM.courant_timestep(model, 0.2)
                SmashMPM.g2p2g!(model, dt)
                SmashMPM.grid_reset!(model.grid)
                model.t += dt
            end
        catch e
            println("\nSimulation interrupted due to an error: ", e)
            break
        end
 
        update_visuals!(vis.position_buffer_3d, vis.position_buffer_slice, vis.color_buffer_slice,
                         model, vis.slice_y_center, vis.slice_thickness,
                         vis.target_color_cairo, vis.projectile_color_cairo)
 
        vis.mass_slice_obs[] = Matrix{Float64}(model.grid.state_old.mass[:, vis.jy, :])
        new_pts, new_dirs, new_speeds = compute_arrow_data(model, vis.jy, vis.arrow_stride,
                                                             vis.mass_threshold, vis.dx_grid)
        vis.arrow_points_obs[] = new_pts
        vis.arrow_dirs_obs[]   = new_dirs
        vis.arrow_speeds_obs[] = new_speeds
 
        notify(vis.points_obs_3d)
        notify(vis.points_obs_slice)
        notify(vis.colors_obs_slice)
 
        last_plot_t = model.t
 
        GLMakie.activate!()
        GLMakie.recordframe!(stream_persp)
 
        CairoMakie.activate!()
        CairoMakie.recordframe!(stream_slice)
 
        if model.t >= model.t_max
            println("\nSimulation completed at frame $frame.")
            break
        end
    end

    GLMakie.destroy!(screen)
 
    return stream_persp, stream_slice
end
 
# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
 
function main()
    dx = 0.04
    body_target, body_projectile, dist, vel_0 = build_bodies()
 
    setup = SimulationSetup(dx=dx, t_max=2 * dist / vel_0 * 5, padding=5, ppc_1d=2,
                             CFL_number=0.4, dt_max=1e-3)
    model = build_mpm_model((body_target, body_projectile), setup)
 
    N_particles = validate_model(model, dx)
 
    grid_dims = size(model.grid.state_old.mass)
    println("Model built successfully with $N_particles particles and a grid of size $grid_dims.")
 
    vis = setup_visualization(model, N_particles, vel_0, dx)
    stream_persp, stream_slice = run_simulation!(model, vis)
 
    GLMakie.save(OUTPUT_PERSPECTIVE, stream_persp)
    CairoMakie.save(OUTPUT_SLICE, stream_slice)
 
    println("Videos successfully saved as '$OUTPUT_PERSPECTIVE' and '$OUTPUT_SLICE'!")
end
 
main()
