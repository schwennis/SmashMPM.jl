struct Particle{T, MatCacheType<:AbstractMaterialCache}
    pos::SVector{3, T}

    m::T
    V0::T

    F::SMatrix{3, 3, T, 9}

    mat_cache::MatCacheType
end


function Particle{T, MatCacheType}(x::SVector{3, T}, m::T, V0::T, mat_cache::MatCacheType) where {T, MatCacheType<:AbstractMaterialCache}
    return Particle{T, MatCacheType}(x, m, V0, one(SMatrix{3, 3, T, 9}), mat_cache)
end
