# ---------------------------------------------------------------------------- #
#                                  Dense Grid                                  #
# ---------------------------------------------------------------------------- #
mutable struct DenseGrid{T, S <: AbstractArray}<:AbstractGrid
    state_read::S            # StructArray of GridNodes (from previous time step)
    state_write::S            # StructArray of GridNodes (for current time step)

    padding::Int            # Padding width

    origin::SVector{3, T}   # Space Coordinates of the grid origin (1,1,1) + padding
    inv_dx::T               # Inverse of grid spacing dx
end


function _allocate_grid_state(backend, ::Type{T}, N::SVector{3,Int}) where {T}
    dims = (N[1], N[2], N[3])

    # Fill on CPU
    mass = zeros(T, dims)
    wave_speed = zeros(T, dims)
    mom_x = zeros(T, dims)
    mom_y = zeros(T, dims)
    mom_z = zeros(T, dims)

    # Named tuple to make single momentum components available via grid.state_write.momentum.x, .y, .z
    momentum = StructArray{SVector{3,T}}((
        x=_to_backend(backend, mom_x), 
        y=_to_backend(backend, mom_y), 
        z=_to_backend(backend, mom_z)
    ))
    return StructArray{GridNode{T}}((
        mass=_to_backend(backend, mass), 
        momentum=momentum, 
        wave_speed=_to_backend(backend, wave_speed)
    ))
end

function DenseGrid(dx::T, N::SVector{3,Int}, origin::SVector{3,T}, padding::Int=2,
                    backend=CPU()) where {T}
    inv_dx = one(T) / dx

    state_read = _allocate_grid_state(backend, T, N)
    state_write = _allocate_grid_state(backend, T, N)

    return DenseGrid{T, typeof(state_read)}(state_read, state_write, padding, origin, inv_dx)
end


function max_wavespeed(grid::DenseGrid)
    return maximum(grid.state_read.wave_speed)
end



function grid_reset!(grid::DenseGrid{T, S}) where {T, S}
    grid.state_read, grid.state_write = grid.state_write, grid.state_read

    # reset grid_new for the next iteration
    fill!(grid.state_write.mass, zero(T))
    fill!(grid.state_write.wave_speed, zero(T))
    fill!(grid.state_write.momentum.x, zero(T))
    fill!(grid.state_write.momentum.y, zero(T))
    fill!(grid.state_write.momentum.z, zero(T))
end

