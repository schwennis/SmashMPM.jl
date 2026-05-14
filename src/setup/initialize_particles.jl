# Pipeline:
# Start: List of positions, velocities, masses, volumes
# generate material groups with pos, mass, volume
# scatter velocities to grid



function add_material_group!(model::Model, positions, masses, volumes, material)
    # Check for NaN values in the input data
    for i in eachindex(positions)
        if any(isnan.(positions[i])) || isnan(masses[i]) || isnan(volumes[i])
            error("NaN values detected in particle data at index $i")
        end
        if any(isinf.(positions[i])) || isinf(masses[i]) || isinf(volumes[i])
            error("Inf values detected in particle data at index $i")
        end
    end

    N = length(positions)
    matcachetype = typeof(MaterialCache(material))
    particle_vector = Vector{Particle{eltype(positions[1]), matcachetype}}(undef, N)
    for i in 1:N
        particle_vector[i] = Particle(positions[i], masses[i], volumes[i], MaterialCache(material))
    end
    material_group = SoAMaterialGroup(particle_vector, material)
    model.material_groups = model.material_groups + (material_group,)
end


