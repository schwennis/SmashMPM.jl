@kwdef struct HypoElasticity{T} <: AbstractElasticity
    μ::T
end

struct HypoElasticState{T}
    s::SMatrix{3,3,T,9}
end

init_state(::HypoElasticity{T}) where {T} = HypoElasticState(zero(SMatrix{3,3,T,9}))
shear_modulus(el::HypoElasticity) = el.μ

@inline function elastic_trial(el::HypoElasticity, st::HypoElasticState, kin::Kinematics)
    D     = (kin.C + kin.C') / 2
    W     = (kin.C - kin.C') / 2
    D_dev = D - one(SMatrix{3,3,eltype(kin.C),9}) * tr(D) / 3
    s     = st.s
    s_tr  = s + kin.dt * (2el.μ * D_dev + W * s - s * W)
    return s_tr, st, el.μ
end


@inline consistent_elastic_state(::HypoElasticity, ::HypoElasticState, s, kin::Kinematics) =
    HypoElasticState(s)