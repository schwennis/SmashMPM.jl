@kwdef struct HyperElasticStrengthModel{T} <: AbstractStrengthModel
    μ::T
end

@kwdef struct HyperElasticStrengthModelState{T} <: AbstractStrengthModelState
end

function init_strength_state(::HyperElasticStrengthModel{T}) where {T}
    return HyperElasticStrengthModelState{T}()
end








function strength_model(strength_model::HyperElasticStrengthModel, strength_state::HyperElasticStrengthModelState, ρ::T, F::SMatrix{3, 3, T, 9}, D_dev::SMatrix{3, 3, T, 9}, W::SMatrix{3, 3, T, 9}, dt::T) where {T}
    μ = strength_model.μ

    J = det(F)
    b = J^T(-2/3) * F * F'

    s_new = μ/J * (b - tr(b)/T(3) * one(SMatrix{3, 3, T, 9}))

    # p-wave speed
    c_p = sqrt(T(4) * μ / (T(3) * ρ))

    return s_new, HyperElasticStrengthModelState{T}(), c_p
end

