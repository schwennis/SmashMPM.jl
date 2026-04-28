struct NeoHookean{T}<:AbstractMaterial
    μ::T
    λ::T
    ρ::T
    c::T
end


function material_model(material::NeoHookean{T}, mat_cache::NoMaterialCache, F, C, V0, m, dt) where {T}
    μ = material.μ
    λ = material.λ

    J = det(F)
    b = F * F'

    I_mat = one(F)

    σ = (μ * (b - I_mat) + λ * log(J) * I_mat) / J

    return σ, mat_cache
end

function get_soundspeed(material::NeoHookean{T}, material_cache::NoMaterialCache) where {T}
    return material.c
end