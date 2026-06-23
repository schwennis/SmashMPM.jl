abstract type AbstractGrid end


struct GridNode{T}
    mass::T
    momentum::SVector{3, T}
    wave_speed::T   # maximum wave speed v_p + c_p around this node (for CFL condition)
end

function Base.zero(::Type{GridNode{T}}) where {T}
    return GridNode(zero(T), zero(SVector{3, T}), zero(T))
end


# ---------------------------------------------------------------------------- #
#                                  Dense Grid                                  #
# ---------------------------------------------------------------------------- #
mutable struct DenseGrid{T, S <: AbstractArray}<:AbstractGrid
    state_old::S            # StructArray of GridNodes (from previous time step)
    state_new::S            # StructArray of GridNodes (for current time step)

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

    # Named tuple to make single momentum components available via grid.state_new.momentum.x, .y, .z
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

    state_old = _allocate_grid_state(backend, T, N)
    state_new = _allocate_grid_state(backend, T, N)

    return DenseGrid{T, typeof(state_old)}(state_old, state_new, padding, origin, inv_dx)
end


function max_wavespeed(grid::DenseGrid{T, S}) where {T, S}
    res = mapreduce(max, grid.state_old) do node
        node.wave_speed
    end
    return res
end



function grid_reset!(grid::DenseGrid{T, S}) where {T, S}
    grid.state_old, grid.state_new = grid.state_new, grid.state_old

    # reset grid_new for the next iteration
    fill!(grid.state_new.mass, zero(T))
    fill!(grid.state_new.wave_speed, zero(T))
    fill!(grid.state_new.momentum.x, zero(T))
    fill!(grid.state_new.momentum.y, zero(T))
    fill!(grid.state_new.momentum.z, zero(T))
end

