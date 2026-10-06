@kwdef struct HyperElasticity{T} <: AbstractElasticity
    μ::T
end

struct HyperElasticState{T}
    bbar_e::SMatrix{3,3,T,9}
end

init_state(::HyperElasticity{T}) where {T} = HyperElasticState(one(SMatrix{3,3,T,9}))
shear_modulus(el::HyperElasticity) = el.μ

@inline function elastic_trial(el::HyperElasticity, st::HyperElasticState{T}, kin::Kinematics{T}) where {T}
    J      = det(kin.F)
    Δf     = one(SMatrix{3,3,T,9}) + kin.dt * kin.C
    Δf_iso = Δf / cbrt(det(Δf))
    b      = Δf_iso * st.bbar_e * Δf_iso'
    b      = (b + b') / 2

    s = el.μ / J * (b - (tr(b) / 3) * I)
    G = el.μ * tr(b) / (3J)          # effective shear modulus μ̄/J
    return s, HyperElasticState(b), G
end


@inline function consistent_elastic_state(el::HyperElasticity, st_tr::HyperElasticState, s, kin::Kinematics)
    Ie = tr(st_tr.bbar_e) / 3
    return HyperElasticState(det(kin.F) / el.μ * s + Ie * I)
end