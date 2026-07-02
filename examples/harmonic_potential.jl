using SmashMPM
using StaticArrays
using LinearAlgebra

dx = 0.1
min_coords = SVector(-1.0, -1.0, -1.0)
max_coords = SVector(1.0, 1.0, 1.0)
Ns = SVector{3, Int}(floor.((max_coords .- min_coords) ./ dx)...)
padding = 0
grid = DenseGrid(dx, Ns, min_coords, padding)


ω = 2π/10
F_0 = ω^2

force_field = Array{SVector{3, Float64}}(undef, (Ns[1], Ns[2], Ns[3]))
ext_force = SmashMPM.VectorFieldForce(force_field)
for i in 1:Ns[1], j in 1:Ns[2], k in 1:Ns[3]
    z = grid.origin[3] + (k - 1) * dx 
    
    force_field[i, j, k] = - z * F_0 * SVector(0.0, 0.0, 1.0) 
end


pos_p_0 = SVector(0.0, 0.0, 0.5)
particle = Particle(1, pos_p_0, 1.0, 1.0, NoMaterialState())
particles = Vector{Particle{Float64, NoMaterialState}}(undef, 1)
particles[1] = particle
particle_set = SoAParticleSet(deepcopy(particles), NoMaterialModel())

model = MPMModel(
    particle_sets=(particle_set,),
    grid=grid,
    t_max = 30.0,
    dt_max = 1e-2
)


z_particles = []
ts = []

while model.t < model.t_max
    dt = SmashMPM.courant_timestep(model, 0.5)
    SmashMPM.g2p2g!(model, dt)
    SmashMPM.grid_reset!(model.grid)
    SmashMPM.apply_external_forces!(ext_force, model.grid, dt)
    model.t += dt
    push!(z_particles, model.particle_sets[1].particles.pos[1][3])
    push!(ts, model.t)
end

z_analytical = 0.5 * cos.(ω .* ts)

log_error = log10.(abs.(z_particles .- z_analytical))


using CairoMakie
fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], xlabel = "Time", ylabel = "Z Position")
scatter!(ax, ts[1:60:end], z_particles[1:60:end], color = :blue, marker = :xcross, label = "MPM Simulation")
lines!(ax, ts, z_analytical, color = :red, label = "Analytical Solution")
axislegend(ax, position = :rb)
save("harmonic_potential.png", fig)

fig = Figure(size = (800, 600))
ax = Axis(fig[1, 1], xlabel = "Time", ylabel = "Log Error")
lines!(ax, ts, log_error, color = :green, label = "Log Error")
axislegend(ax, position = :rb)
save("harmonic_potential_error.png", fig)




