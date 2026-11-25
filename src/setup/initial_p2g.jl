# ---------------------------------------------------------------------------- #
#                                  Initial P2G                                 #
# ---------------------------------------------------------------------------- #
# Scatters mass and momentum (no stress yet) of the initial particle state into
# grid.state_write. One kernel for all grid types via the grid access interface
# (src/grids/grid_access.jl). On a SparseGrid the write tiles must have been
# activated before (prepare_initial_p2g!); finalize_initial_p2g! swaps write -> read.
@inline function initial_p2g_particle!(p_idx, grid, positions, velocities, affine, masses,
                                       origin, inv_dx, spline)
    pos  = positions[p_idx]
    vel  = velocities[p_idx]
    mass = masses[p_idx]

    grid_pos = grid_position(pos, inv_dx, origin)
    if !position_ok(grid, grid_pos)     # SparseGrid: NaN or outside the key range
        record_miss!(grid)
        return nothing
    end

    base_node = get_support_base(spline, grid_pos)
    cache     = stencil_cache(grid, spline, base_node)
    iterator_i, iterator_j, iterator_k = get_support_offsets(spline)

    for di in iterator_i, dj in iterator_j, dk in iterator_k
        i = base_node[1] + di
        j = base_node[2] + dj
        k = base_node[3] + dk

        idx, status = node_slot(grid, cache, i, j, k)
        if status != NODE_OK
            # outside the grid / box: clipped; inactive tile: lost write
            status == NODE_INACTIVE && record_miss!(grid)
            continue
        end

        natural_coords = grid_pos - SVector(i, j, k)
        N = shapefunction(spline, natural_coords)
        r_rel = - natural_coords * (1 / inv_dx)

        p_update = N * (mass * vel + mass * affine * r_rel) # Stress isnt calculated yet, so we need the factor m
        scatter!(grid, idx, N * mass, p_update)
    end
    return nothing
end

@kernel function initial_p2g_kernel!(grid, positions, velocities, affine, masses, origin, inv_dx, spline)
    p_idx = @index(Global, Linear)
    initial_p2g_particle!(p_idx, grid, positions, velocities, affine, masses, origin, inv_dx, spline)
end

"""
    initial_p2g!(grid, positions, velocities, affine, masses, spline)

P2G of one body (common affine matrix `affine`) into `grid.state_write`.
"""
function initial_p2g!(grid::AbstractGrid, positions, velocities, affine, masses, spline)
    isempty(positions) && return nothing
    check_stencil(grid, spline)

    backend = KernelAbstractions.get_backend(grid.state_write.mass)
    kernel  = initial_p2g_kernel!(backend)

    kernel(write_access(grid), positions, velocities, affine, masses, grid.origin, grid.inv_dx, spline;
           ndrange=length(positions))

    KernelAbstractions.synchronize(backend)
    return nothing
end


# ---------------------------------------------------------------------------- #
#                       Preparation / finalization per grid                    #
# ---------------------------------------------------------------------------- #
# DenseGrid: initial_p2g! writes into state_write, afterwards read and write states are swapped
prepare_initial_p2g!(::DenseGrid, particle_sets, spline)  = nothing
finalize_initial_p2g!(grid::DenseGrid, particle_sets, spline) = grid_reset!(grid)

# SparseGrid: activate the write tiles, scatter into them, then swap them to read
prepare_initial_p2g!(grid::SparseGrid, particle_sets, spline) = (activate_tiles!(grid, particle_sets, spline); nothing)
finalize_initial_p2g!(grid::SparseGrid, particle_sets, spline) = grid_reset!(grid, particle_sets, spline)