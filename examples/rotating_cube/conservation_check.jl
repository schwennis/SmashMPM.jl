using HDF5
using Printf
using StaticArrays
using LinearAlgebra
using CairoMakie
using Glob

function extract_energy_and_angular_momentum(file_path::String)
    h5open(file_path, "r") do file
        t = attrs(file)["time"]
        masses = read(file["mass"])
        T = eltype(masses)
        masses = Float64.(masses)
        
        positions_mat = read(file["position"])
        positions = reinterpret(reshape, SVector{3, T}, positions_mat)
        positions = SVector{3, Float64}.(positions)

        velocities_mat = read(file["velocity"])
        velocities = reinterpret(reshape, SVector{3, T}, velocities_mat)
        velocities = SVector{3, Float64}.(velocities)

        total_mass = sum(masses)
        COM = sum(masses .* positions) / total_mass

        # Ref(COM) verhindert den DimensionMismatch beim Broadcasting
        positions_com = positions .- Ref(COM)

        # Kinetische Energie: 0.5 * sum(m * ||v||^2)
        kinetic_energy = 0.5 * sum(m * dot(v, v) for (m, v) in zip(masses, velocities))

        # Gesamtdrehimpuls-Vektor: L = sum(m * (r x v))
        angular_momentum_vec = sum(m * cross(r, v) for (m, r, v) in zip(masses, positions_com, velocities))

        # Falls der Betrag geplottet werden soll (norm)
        angular_momentum = norm(angular_momentum_vec)

        return t, kinetic_energy, angular_momentum
    end
end

function main()
    files = glob("output_F64_NeoHookean/rotating_cube*.h5")
    sort!(files, by = f -> parse(Int, match(r"\d+", basename(f)).match))

    times = Float64[]
    kinetic_energies = Float64[]
    angular_momenta = Float64[]

    for (i, file_path) in enumerate(files)
        print("Processing $(i)-th file: $file_path          \r")
        t, kinetic_energy, angular_momentum = extract_energy_and_angular_momentum(file_path)
        if t < 5
            continue
        end
        push!(times, t)
        push!(kinetic_energies, kinetic_energy)
        push!(angular_momenta, angular_momentum)
    end
    println()

    # Normalisierung auf den Startwert (Schritt 1)
    kinetic_energies_rel = kinetic_energies ./ kinetic_energies[1]
    angular_momenta_rel = angular_momenta ./ angular_momenta[1]

    println("Final deviation in kinetic energy: $(abs(kinetic_energies_rel[end] - 1))")
    println("Final deviation in angular momentum: $(abs(angular_momenta_rel[end] - 1))")

    # Plotting
    fig = Figure(size = (800, 400))
    
    ax1 = Axis(fig[1, 1], xlabel = "Time t", ylabel = "E_kin / E_kin(0)")
    lines!(ax1, times, kinetic_energies_rel, color = :blue, label = "Kinetic Energy")
    axislegend(ax1)

    ax2 = Axis(fig[1, 2], xlabel = "Time t", ylabel = "L / L(0)")
    lines!(ax2, times, angular_momenta_rel, color = :red, label = "Angular Momentum")
    axislegend(ax2)

    save("energy_and_angular_momentum.png", fig)
end

main()