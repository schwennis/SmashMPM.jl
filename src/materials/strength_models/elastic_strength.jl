@kwdef struct ElasticStrengthModel{T} <: AbstractStrengthModel
    μ::T
end

@kwdef struct ElasticStrengthModelState{T} <: AbstractStrengthModelState
    s::SMatrix{3, 3, T, 9}  # Deviatoric stress tensor 
end

function init_strength_state(::ElasticStrengthModel{T}) where {T}
    return ElasticStrengthModelState{T}(zeros(SMatrix{3,3,T,9}))
end





function strength_model(strength_model::ElasticStrengthModel, strength_state::ElasticStrengthModelState, ρ::T, D_dev::SMatrix{3, 3, T, 9}, W::SMatrix{3, 3, T, 9}, dt::T) where {T}
    μ = strength_model.μ
    s_old = strength_state.s

    # Rotate the stress tensor
    s_rot = W * s_old - s_old * W

    # Update the deviatoric stress tensor
    s_new = s_old + dt * (2 * μ * D_dev + s_rot)

    # p-wave speed
    c_p = sqrt(T(4) * μ / (T(3) * ρ))

    return s_new, ElasticStrengthModelState{T}(s_new), c_p
end

