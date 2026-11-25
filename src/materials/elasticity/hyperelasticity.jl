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


# b̄ₑ = dev(b̄ₑ) + Iₑ I with dev(b̄ₑ) = J s / μ from the return map. Iₑ is chosen such that
# det b̄ₑ = 1 (Simo & Hughes 1998, remark to Box 9.1), otherwise μ/2 (tr b̄ₑ - 3) would not be the
# stored energy after plastic flow (keeping tr b̄ₑ fixed gives W_{n+1} = W_trial -> negative heat).
# det(A + x I) = x³ + I₂ x + I₃ for deviatoric A; Newton from the trial value (exact root in
# elastic steps, otherwise the start is off by O(Δεₚ²) -> 3 iterations are plenty).
@inline function consistent_elastic_state(el::HyperElasticity, st_tr::HyperElasticState{T}, s, kin::Kinematics) where {T}
    A  = det(kin.F) / el.μ * s
    I2 = -tr(A * A) / 2
    I3 = det(A)
    x  = tr(st_tr.bbar_e) / 3
    for _ in 1:3
        x -= (x^3 + I2 * x + I3 - one(T)) / (3x^2 + I2)
    end
    return HyperElasticState(A + x * I)
end

# Isochoric neo-Hooke: τ' = μ dev(b̄ₑ),  W_el = μ/2 (tr b̄ₑ - 3) per reference volume
@inline function kirchhoff_deviator(el::HyperElasticity, st::HyperElasticState, J)
    b = st.bbar_e
    return el.μ * (b - (tr(b) / 3) * I)
end
@inline elastic_energy(el::HyperElasticity, st::HyperElasticState, J) = el.μ / 2 * (tr(st.bbar_e) - 3)