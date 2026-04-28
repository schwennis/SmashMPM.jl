struct SoAMaterialGroup{MaterialType<:AbstractMaterial, SA<:StructArray}<:AbstractMaterialGroup
    points::SA              # StructArray{Particle{T,MaterialCache}}
    points_buffer::SA       # Buffer for fast swapping of points (StructArray{Particle{T,MaterialCache}})
    material::MaterialType
end

function SoAMaterialGroup(particle_vector::AbstractVector{Particle{T, MatCacheType}}, material::MaterialType, ::Type{AT}=Array) where {T, MatCacheType, MaterialType <: AbstractMaterial, AT<:AbstractArray}
    
    get_device_SoA() = begin
        cpu_SoA = StructArray{Particle{T, MatCacheType}}(
            undef, Tuple(length(particle_vector));
            unwrap = t -> t <: SVector
        )
        fill!(cpu_SoA, zero(Particle{T, MatCacheType}))
        return StructArrays.replace_storage(AT, cpu_SoA)
    end

    SoA = get_device_SoA()
    SoA_buffer = get_device_SoA()
    return SoAMaterialGroup{MaterialType, typeof(SoA)}(SoA, SoA_buffer, material)
end