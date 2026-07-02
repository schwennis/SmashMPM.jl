using SmashMPM
using StaticArrays
using GLMakie
using FFMPEG

# Target Parameters
R_target = 0.5
center_target = SVector{3, Float64}(0.0, 0.0, 0.0)
mat_target = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
shape_target = SmashMPM.Sphere(center=center_target, radius=R_target)
body_target = Body(shape_target, zero(center_target), zero(center_target), mat_target)

# Projectile Parameters
R_projectile = 0.2
dist = R_target + R_projectile + 1.0
impact_parameter = 0.0
mat_projectile = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
vel_0 = 0.1

v_projectile = vel_0 * SVector(-1.0, 0.0, 0.0)
center_projectile = center_target + SVector(dist, impact_parameter, 0.0)
shape_projectile = SmashMPM.Sphere(center=center_projectile, radius=R_projectile)
body_projectile = Body(shape_projectile, v_projectile, zero(v_projectile), mat_projectile)

dx = 0.05
Setup = SimulationSetup(dx=0.05, t_max=2*dist/vel_0, padding=4, ppc_1d=2, CFL_number=0.4, dt_max=1e-3)
model = build_mpm_model((body_target, body_projectile), Setup)

particle_sets = model.particle_sets
@assert particle_sets[1].particles.F[1] == one(SMatrix{3,3,Float64,9}) "Initial deformation gradient for target is not identity"
@assert particle_sets[2].particles.F[1] == one(SMatrix{3,3,Float64,9}) "Initial deformation gradient for projectile is not identity"
@assert all(particle_sets[1].particles.mass .> 0.0) "Mass for target particles is not positive"
@assert all(particle_sets[2].particles.mass .> 0.0) "Mass for projectile particles is not positive"
@assert all(particle_sets[1].particles.initial_volume .> 0.0) "Volume for target particles is not positive"
@assert all(particle_sets[2].particles.initial_volume .> 0.0) "Volume for projectile particles is not positive"
@assert particle_sets[1].particles.initial_volume[1] ≈ (dx/2)^3 "Initial volume for target particles is not correct"
@assert particle_sets[2].particles.initial_volume[1] ≈ (dx/2)^3 "Initial volume for projectile particles is not correct"
@assert particle_sets[1].particles.mass[1] / particle_sets[1].particles.initial_volume[1]  ≈ particle_sets[1].material.ρ "Volume and mass for target particles do not match"


N_particles = length(model.particle_sets[1].particles.pos) + length(model.particle_sets[2].particles.pos)
dims = size(model.grid.state_old.mass)
println("Model built successfully with $N_particles particles and a grid of size $dims.")

# --- GLMakie Visualisierungs-Setup ---
# Wir extrahieren die initialen Positionen aller Partikel für den Plot
# Konvertierung mittels speichereffizientem cast/reinterpret
# Einmalig ein flaches Array in der richtigen Größe und im richtigen Format allokieren
buffer_positionen = Vector{Point3f}(undef, N_particles)

function update_positions!(buffer, model)
    idx = 1
    
    # Schleife über beide Partikel-Sets (Target und Projectile)
    for set in model.particle_sets
        # Zugriff auf die separaten x, y, z Komponenten des StructArrays
        px = set.particles.pos.x
        py = set.particles.pos.y
        pz = set.particles.pos.z
        
        @inbounds for i in eachindex(px)
            # Direktes Hineinschreiben als Point3f (konvertiert T zu Float32)
            buffer[idx] = Point3f(px[i], py[i], pz[i])
            idx += 1
        end
    end
end
update_positions!(buffer_positionen, model)
points_obs = Observable(buffer_positionen)

# Erstelle die Figure und die 3D-Achse
fig = Figure(size = (800, 600))
ax = Axis3(fig[1, 1], title = "MPM Aufprall-Simulation", aspect = :data)

# Partikel plotten (Farbe kann je nach Set angepasst werden, hier einfach blau/rot getrennt)
colors = vcat(fill(:blue, length(model.particle_sets[1].particles.pos)), 
              fill(:red, length(model.particle_sets[2].particles.pos)))

meshscatter!(ax, points_obs, markersize = 0.02, color = colors)

# Grenzen der Achsen fixieren, damit das Bild stabil bleibt
xlims!(ax, -1.0, 2.5)
ylims!(ax, -1.0, 1.0)
zlims!(ax, -1.0, 1.0)

# Zeige das Fenster an
display(fig)

# --- Simulations-Schleife ---
plot_interval = 0.02
last_plot_t = 0.0

println("Starte Simulation und Video-Aufnahme...")

# Das @record Makro speichert die Animation als "mp4" ab
record(fig, "mpm_aufprall.mp4", frame_iterator = 1:2000) do frame
    global last_plot_t
    # Wir simulieren so lange in Echtzeit weiter, bis das nächste Plot-Intervall erreicht ist
    while model.t < model.t_max && (model.t - last_plot_t < plot_interval)
        dt = SmashMPM.courant_timestep(model, 0.5)
        SmashMPM.g2p2g!(model, dt)
        SmashMPM.grid_reset!(model.grid)
        # SmashMPM.apply_external_forces!(ext_force, model.grid, dt)
        model.t += dt
    end
    
    # Aktualisiere die Positionen für den aktuellen Frame im Video
    update_positions!(buffer_positionen, model)
    points_obs[] = buffer_positionen
    last_plot_t = model.t
    sleep(0.001)
    
    # Wenn die Simulation das Ende erreicht hat, brechen wir die Video-Frames ab
    if model.t >= model.t_max
        # Beendet die Schleife vorzeitig, falls t_max vor Frame 2000 erreicht wird
    end
end

println("Video erfolgreich als 'mpm_aufprall.mp4' gespeichert!")