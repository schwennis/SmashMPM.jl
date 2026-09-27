struct ElasticStrengthModel{T} <: AbstractStrengthModel
    μ::T
end

struct ElasticStrengthModelState{T} <: AbstractStrengthModelState
    s::SMatrix{3, 3, T, 9}  # Deviatoric stress tensor 
end



function strength_model(strength_model::ElasticStrengthModel, strength_state::ElasticStrengthModelState, D_dev, W , dt)
    μ = strength_model.μ
    s_old = strength_state.s

    # Rotate the stress tensor
    s_rot = W * s_old - s_old * W

    # Update the deviatoric stress tensor
    s_new = s_old + dt * (2 * μ * D_dev + s_rot)

    return s_new, ElasticStrengthModelState{typeof(μ)}(s_new)
end