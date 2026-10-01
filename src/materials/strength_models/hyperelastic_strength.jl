@kwdef struct HyperElasticStrengthModel{T} <: AbstractStrengthModel
    μ::T
end

@kwdef struct HyperElasticStrengthModelState{T} <: AbstractStrengthModelState
    b::SMatrix{3, 3, T, 9}  # Right Cauchy-Green deformation tensor
end

function init_strength_state(::HyperElasticStrengthModel{T}) where {T}
    return HyperElasticStrengthModelState{T}(one(SMatrix{3, 3, T, 9}))
end








function strength_model(strength_model::HyperElasticStrengthModel, strength_state::HyperElasticStrengthModelState, ρ::T, F::SMatrix{3, 3, T, 9}, D_dev::SMatrix{3, 3, T, 9}, W::SMatrix{3, 3, T, 9}, dt::T) where {T}
    μ = strength_model.μ

    J = det(F)
    b = J^(T(-2)/3) * F * F'
    b = T(0.5) * (b + b')  # Ensure symmetry

    s_new = μ/J * (b - tr(b)/T(3) * one(SMatrix{3, 3, T, 9}))

    # Deviatoric work
    dev_work_rate = T(0.5) * μ * (tr(b) - tr(strength_state.b)) / dt

    

    # p-wave speed
    c_p = sqrt(T(4) * μ / (T(3) * ρ))

    return s_new, HyperElasticStrengthModelState{T}(b), c_p, dev_work_rate 
end

