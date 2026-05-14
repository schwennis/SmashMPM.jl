struct NeoHookean{T}<:AbstractMaterial
    μ::T
    λ::T
    ρ::T
    c::T
end

function NeoHookean(;E, ν, ρ, λ, μ)
    if isnothing(ρ)
        error("Density ρ must be provided")
    end

    if !isnothing(E) && !isnothing(ν)
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2 * (1 + ν))
    elseif λ === nothing || μ === nothing
        error("Either (E and ν) or (λ and μ) must be provided")
    end

    return NeoHookean{typeof(λ)}(μ, λ, ρ, c)
end

function MaterialCache(::NeoHookean)
    return NoMaterialCache()
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