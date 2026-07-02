# ---------------------------------------------------------------------------- #
#                                Linear Elastic                                #
# ---------------------------------------------------------------------------- #
struct LinearElastic{T}<:AbstractMaterial
    μ::T
    λ::T
    ρ::T
    c::T
end

function LinearElastic(;E=nothing, ν=nothing, ρ=nothing, λ=nothing, μ=nothing)
    if isnothing(ρ)
        error("Density ρ must be provided")
    end

    if !isnothing(E) && !isnothing(ν)
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2 * (1 + ν))
    elseif λ === nothing || μ === nothing
        error("Either (E and ν) or (λ and μ) must be provided")
    end
    c = sqrt((λ + 2 * μ) / ρ)

    return LinearElastic{typeof(λ)}(μ, λ, ρ, c)
end

function get_initial_material_state(::LinearElastic)
    return NoMaterialState()
end


@inline function material_model(material::LinearElastic{T}, mat_cache::NoMaterialState, F, C, V0, m, dt) where {T}
    μ = material.μ
    λ = material.λ

    I = one(SMatrix{3, 3, T, 9})
    
    ε = (F + F') / 2 - I
    
    σ = 2 * μ * ε + λ * tr(ε) * I

    return σ, mat_cache
end

function get_soundspeed(material::LinearElastic{T}, material_cache::NoMaterialState) where {T}
    return material.c
end