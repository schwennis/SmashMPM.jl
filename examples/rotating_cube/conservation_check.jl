using HDF5
using Printf
using StaticArrays
using LinearAlgebra
using CairoMakie
using Glob

function extract_energy_and_particle_angular_momentum(file_path::String)
    h5open(file_path, "r") do file
        t = attrs(file)["time"]
        
        # 1. Partikel einlesen
        masses = Float64.(read(file["mass"]))
        total_particles = length(masses)
        
        pos_mat = read(file["position"])
        T_pos = eltype(pos_mat)
        particle_positions = SVector{3, Float64}.(reinterpret(reshape, SVector{3, T_pos}, pos_mat))

        vel_mat = read(file["velocity"])
        T_vel = eltype(vel_mat)
        velocities = SVector{3, Float64}.(reinterpret(reshape, SVector{3, T_vel}, vel_mat))

        # 2. Grid einlesen
        grid_origin_raw = read(file["grid/origin"])
        dx = Float64(read(file["grid/dx"]))
        grid_masses = Float64.(read(file["grid/mass"]))
        
        grid_mom_mat = read(file["grid/momentum"])
        T_mom = eltype(grid_mom_mat)
        grid_momentum = SVector{3, Float64}.(reinterpret(reshape, SVector{3, T_mom}, grid_mom_mat))

        # Falls grid/origin noch nicht um padding bereinigt wurde (z.B. padding = 2):
        # padding = 2
        # origin = SVector{3, Float64}(grid_origin_raw) .- padding * dx
        origin = SVector{3, Float64}(grid_origin_raw)

        dims = size(grid_masses)
        grid_positions = [origin + dx * SVector{3, Float64}(i, j, k) 
                          for i in 0:dims[1]-1, j in 0:dims[2]-1, k in 0:dims[3]-1]

        # 3. Massen und Schwerpunkte
        particle_total_mass = sum(masses)
        particle_COM = sum(masses .* particle_positions) / particle_total_mass

        grid_total_mass = sum(grid_masses)
        grid_COM = sum(grid_masses .* grid_positions) / grid_total_mass

        particle_positions_com = particle_positions .- Ref(particle_COM)
        grid_positions_com = grid_positions .- Ref(grid_COM)

        # 4. Kinetische Energien (E_kin = 0.5 * m * v^2 bzw. 0.5 * p^2 / m)
        kinetic_energy = 0.5 * sum(m * dot(v, v) for (m, v) in zip(masses, velocities))
        grid_kinetic_energy = 0.5 * sum(m > 0.0 ? dot(p, p) / m : 0.0 for (m, p) in zip(grid_masses, grid_momentum))

        # 5. Drehimpulse (L = m * (r x v) bzw. r x p)
        particle_angular_momentum_vec = sum(m * cross(r, v) for (m, r, v) in zip(masses, particle_positions_com, velocities))
        grid_angular_momentum_vec = sum(cross(r, p) for (r, p) in zip(grid_positions_com, grid_momentum))

        return t, kinetic_energy, norm(particle_angular_momentum_vec), grid_kinetic_energy, norm(grid_angular_momentum_vec)
    end
end

function main()
    files = glob("output_analytic_hyperelastic/rotating_cube*.h5")
    sort!(files, by = f -> parse(Int, match(r"\d+", basename(f)).match))

    times = Float64[]
    particle_kinetic_energies = Float64[]
    particle_angular_momenta = Float64[]
    grid_kinetic_energies = Float64[]
    grid_angular_momenta = Float64[]

    for (i, file_path) in enumerate(files)
        print("Processing $(i)-th file: $file_path          \r")
        t, particle_kinetic_energy, particle_angular_momentum, grid_kinetic_energy, grid_angular_momentum = extract_energy_and_particle_angular_momentum(file_path)
        if i == 1
            continue
        end
        push!(times, t)
        push!(particle_kinetic_energies, particle_kinetic_energy)
        push!(particle_angular_momenta, particle_angular_momentum)
        push!(grid_kinetic_energies, grid_kinetic_energy)
        push!(grid_angular_momenta, grid_angular_momentum)
    end
    println()

    # Normalisierung auf den Startwert (Schritt 1)
    particle_kinetic_energies_rel = particle_kinetic_energies ./ particle_kinetic_energies[1]
    particle_angular_momenta_rel = particle_angular_momenta ./ particle_angular_momenta[1]
    grid_kinetic_energies_rel = grid_kinetic_energies ./ grid_kinetic_energies[1]
    grid_angular_momenta_rel = grid_angular_momenta ./ grid_angular_momenta[1]

    println("Particles: Final deviation in kinetic energy: $(abs(particle_kinetic_energies_rel[end] - 1))")
    println("Particles: Final deviation in particle_angular momentum: $(abs(particle_angular_momenta_rel[end] - 1))")
    println("Grid: Final deviation in kinetic energy: $(abs(grid_kinetic_energies_rel[end] - 1))")
    println("Grid: Final deviation in angular momentum: $(abs(grid_angular_momenta_rel[end] - 1))")

    # Plotting
    fig = Figure(size = (800, 400))
    
    ax1 = Axis(fig[1, 1], xlabel = "Time t", ylabel = "E_kin / E_kin(0)")
    lines!(ax1, times, particle_kinetic_energies_rel, color = :blue, label = "Kinetic Energy")
    axislegend(ax1)

    ax2 = Axis(fig[1, 2], xlabel = "Time t", ylabel = "L / L(0)")
    lines!(ax2, times, particle_angular_momenta_rel, color = :red, label = "particle_angular Momentum")
    axislegend(ax2)

    ax3 = Axis(fig[2, 1], xlabel = "Time t", ylabel = "E_kin / E_kin(0)")
    lines!(ax3, times, grid_kinetic_energies_rel, color = :green, label = "Grid Kinetic Energy")
    axislegend(ax3)

    ax4 = Axis(fig[2, 2], xlabel = "Time t", ylabel = "L / L(0)")
    lines!(ax4, times, grid_angular_momenta_rel, color = :orange, label = "Grid Angular Momentum")
    axislegend(ax4)

    save("energy_and_particle_angular_momentum.png", fig)
end

main()