function autogrid_parameters!(positions; fixed_particle_distance=nothing, ppc_1d=2, cells_to_border=4, padding=2, max_dx=nothing, min_dx=nothing)
    @assert ppc_1d > 0 "Particles per cell (ppc_1d) must be greater than 0"
    if max_dx !== nothing
        @assert max_dx > 0 "max_dx must be greater than 0"
    end
    if min_dx !== nothing
        @assert min_dx > 0 "min_dx must be greater than 0"
    end
    if max_dx !== nothing && min_dx !== nothing
        @assert max_dx >= min_dx "max_dx must be greater than or equal to min_dx"
    end

    # Find dx
 
    if isnothing(fixed_particle_distance)
        # println("No fixed_particle_distance provided, estimating from particle positions...")
        dx_particles = median_nn_spacing(positions)
    else
        # println("Using provided fixed_particle_distance: $(fixed_particle_distance)")
        dx_particles = fixed_particle_distance
    end
    # println("Found particle spacing: $(dx_particles)")

    dx = dx_particles * ppc_1d

    if max_dx !== nothing
        dx = min(dx, max_dx)
    end
    if min_dx !== nothing
        dx = max(dx, min_dx)
    end

    inv_dx = 1.0 / dx

    min_pos = positions[1]
    max_pos = positions[1]
    for p in positions
        min_pos = min.(min_pos, p)
        max_pos = max.(max_pos, p)
    end

    # Compute the bounding box of the particle positions
    origin = min_pos .- cells_to_border * dx
    max_pos_padded = max_pos .+ cells_to_border * dx

    # --- OPTIMIERUNG 2: Vektorisierte Berechnung für N ---
    N_vec = ceil.(Int, (max_pos_padded .- origin) .* inv_dx) .+ (2 * padding)


    return dx, N_vec, SVector(origin...), padding
    
end



function median_nn_spacing(positions::AbstractVector{SVector{3,T}}) where T
    X = hcat(positions...)
    tree = BallTree(X)
    ds, _ = knnquery(tree, X, 2)
    return median(ds[2, :])
end