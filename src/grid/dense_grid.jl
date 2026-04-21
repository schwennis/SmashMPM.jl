struct DenseGrid{T, S <: AbstractArray}<:AbstractGrid
    state_old::S            # StructArray of GridNodes (from previous time step)
    state_new::S            # StructArray of GridNodes (for current time step)

    padding::Int    # Padding width

    origin::SVector{3, T}           # Space Coordinates of the grid origin (1,1,1)
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


function max_v_c(grid::DenseGrid{T, S}) where {T, S}
    res = mapreduce(max, grid.state_old) do node
        sqrt(sum(node.v.^2)) + node.c
    end
    return res
end