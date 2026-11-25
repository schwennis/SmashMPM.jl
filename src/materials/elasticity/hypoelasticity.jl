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

# s is the Cauchy deviator -> τ' = J s, stored energy J s:s / (4μ) per reference volume
@inline kirchhoff_deviator(::HypoElasticity, st::HypoElasticState, J) = J * st.s
@inline elastic_energy(el::HypoElasticity, st::HypoElasticState, J) =
    iszero(el.μ) ? zero(J) : J * dot(st.s, st.s) / (4el.μ)