struct DenseGrid{T, S <: AbstractArray}<:AbstractGrid
    state_old::S            # StructArray of GridNodes (from previous time step)
    state_new::S            # StructArray of GridNodes (for current time step)

    padding::Int    # Padding width

    origin::SVector{3, T}           # Space Coordinates of the grid origin (1,1,1) + padding
    inv_dx::T                       # Inverse of grid spacing dx
end


function DenseGrid(dx::T, N::SVector{3, Int}, origin::SVector{3, T}, padding::Int=2, ::Type{AT}=Array) where {T, AT<:AbstractArray}
    inv_dx = one(T)/dx

    get_device_state() = begin
        cpu_state = StructArray{GridNode{T}}(
            undef, Tuple(N);
            unwrap = t -> t <: SVector
        )
        fill!(cpu_state, zero(GridNode{T}))
        return StructArrays.replace_storage(AT, cpu_state)
    end

    return DenseGrid(get_device_state(), get_device_state(), padding, origin, inv_dx)
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
    fill!(grid.state_new.m, zero(T))
    fill!(grid.state_new.wave_speed, zero(T))
    fill!(grid.state_new.p.x, zero(T))
    fill!(grid.state_new.p.y, zero(T))
    fill!(grid.state_new.p.z, zero(T))
end