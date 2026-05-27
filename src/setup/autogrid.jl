function autogrid_parameters!(positions, ppc_1d=2; cells_to_border=4, padding=2, max_dx, min_dx)
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
 
    dx_particles = median_nn_spacing(positions)
    dx = dx_particles * ppc_1d

    if max_dx !== nothing
        dx = min(dx, max_dx)
    end
    if min_dx !== nothing
        dx = max(dx, min_dx)
    end

    # Compute the bounding box of the particle positions
    origin = minimum(positions, dims=1) .- cells_to_border * dx
    max_pos = maximum(positions, dims=1) .+ cells_to_border * dx

    Nx = ceil(Int, (max_pos[1] - origin[1]) * inv_dx) + 2 * padding
    Ny = ceil(Int, (max_pos[2] - origin[2]) * inv_dx) + 2 * padding
    Nz = ceil(Int, (max_pos[3] - origin[3]) * inv_dx) + 2 * padding
    

    return dx, SVector(Nx, Ny, Nz), SVector(origin...), padding
    
end



function median_nn_spacing(positions::AbstractVector{SVector{3,<:Number}})
    X = hcat(positions...)
    tree = BallTree(X)
    ds, _ = knnquery(tree, X, 2)
    return median(ds[2, :])
end