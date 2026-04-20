struct DenseGrid{T, S <: AbstractArray}<:AbstractGrid
    state::S            # StructArray of GridNodes

    padding::Int    # Padding width

    origin::SVector{3, T}           # Space Coordinates of the grid origin (1,1,1)
    inv_dx::T                       # Inverse of grid spacing dx
end


function DenseGrid(dx::T, N::SVector{3, Int}, origin::SVector{3, T}, padding::Int=2, ::Type{AT}=Array) where {T, AT<:AbstractArray}
    inv_dx = one(T)/dx

    cpu_state = StructArray{GridNode{T}}(
        undef, Tuple(N);
        unwrap = t -> t <: SVector
    )

    fill!(cpu_state, zero(GridNode{T}))

    device_state = StructArrays.replace_storage(AT, cpu_state)

    return DenseGrid(device_state, padding, origin, inv_dx)
end
