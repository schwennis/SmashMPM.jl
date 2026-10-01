@kwdef struct HypoElasticStrengthModel{T} <: AbstractStrengthModel
    μ::T
end

@kwdef struct HypoElasticStrengthModelState{T} <: AbstractStrengthModelState
    s::SMatrix{3, 3, T, 9}  # Deviatoric stress tensor 
end

function init_strength_state(::HypoElasticStrengthModel{T}) where {T}
    return HypoElasticStrengthModelState{T}(zeros(SMatrix{3,3,T,9}))
end






function strength_model(strength_model::HypoElasticStrengthModel, strength_state::HypoElasticStrengthModelState, ρ::T, _F::SMatrix{3, 3, T, 9}, D_dev::SMatrix{3, 3, T, 9}, W::SMatrix{3, 3, T, 9}, dt::T) where {T}
    μ = strength_model.μ
    s_old = strength_state.s

    # Rotate the stress tensor
    s_rot = W * s_old - s_old * W

    # Update the deviatoric stress tensor
    s_new = s_old + dt * (2 * μ * D_dev + s_rot)

    # Compute the deviatoric work rate
    dev_work_rate = dot(s_new, D_dev)

    # p-wave speed
    c_p = sqrt(T(4) * μ / (T(3) * ρ))

    return s_new, HypoElasticStrengthModelState{T}(s_new), c_p, dev_work_rate
end

