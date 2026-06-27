# ---------------------------------------------------------------------------- #
#                                  NeoHookean                                  #
# ---------------------------------------------------------------------------- #
struct NeoHookean{T}<:AbstractMaterial
    μ::T
    λ::T
    ρ::T
    c::T
end

function NeoHookean(;E=nothing, ν=nothing, ρ=nothing, λ=nothing, μ=nothing)
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

    return NeoHookean{typeof(λ)}(μ, λ, ρ, c)
end

function get_initial_material_state(::NeoHookean)
    return NoMaterialState()
end


@inline function material_model(material::NeoHookean{T}, mat_state::NoMaterialState, F, C, V0, m, dt) where {T}
    μ = material.μ
    λ = material.λ

    @fastmath J = det(F)    # For some reason @fastmath causes a big slowdown everywhere else
    b = F * F'
    I = one(SMatrix{3,3,T,9})

    σ = (μ * (b - I) + λ * log(J) * I) / J

    return σ, mat_state
end

function get_soundspeed(material::NeoHookean{T}, material_cache::NoMaterialState) where {T}
    return material.c
end