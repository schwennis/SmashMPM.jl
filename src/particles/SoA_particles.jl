struct SoAMaterialGroup{MaterialType<:AbstractMaterial, SA<:StructArray}
    points::SA              # StructArray{Particle{T,MaterialCache}}
    material::MaterialType
end

function SoAMaterialGroup(
        particle_vector::AbstractVector{Particle{T, MatCacheType}}, 
        material::MaterialType
    ) where {T, MatCacheType, MaterialType <: AbstractMaterial}
    
    SoA = StructArray(particle_vector)
    return SoAMaterialGroup{T, MaterialType, typeof(SoA)}(SoA, material)
end