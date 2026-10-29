# ---------------------------------------------------------------------------- #
#                                 J2Plasticity                                 #
# ---------------------------------------------------------------------------- #
# As described in Simo and Hughes, "Computational Inelasticity", 1998, Section 2.2
@kwdef struct J2Plasticity{T<:Real} <: AbstractPlasticity
    σ_y0::T             # initial yield stress
    H::T = zero(σ_y0)   # linear hardening modulus
end

struct J2PlasticState{T}
    eps_p::T             # equivalent plastic strain
end


init_state(::J2Plasticity{T}) where {T} = J2PlasticState(zero(T))
equivalent_plastic_strain(st::J2PlasticState) = st.eps_p
damage_variable(::J2PlasticState) = false



@inline function return_map(pl::J2Plasticity, st::J2PlasticState, s_tr::SMatrix{3,3,T,9},
                            μ, _th, _kin::Kinematics{T}) where {T}
    # Material parameters in textbook notation, in the precision of the kinematics
    σ_Y = T(pl.σ_y0)                    # initial yield stress
    K′  = T(pl.H)                       # isotropic hardening modulus
    α_n = T(st.eps_p)                   # equivalent plastic strain at t_n
    sqrt_2_3 = sqrt(T(2) / T(3))

    # 1. Check yield condition at the trial state
    norm_s_tr = norm(s_tr)
    f_tr = norm_s_tr - sqrt_2_3 * (σ_Y + K′ * α_n)

    if f_tr <= zero(T)
        return s_tr, st                 # elastic step: (•)_{n+1} = (•)_tr, Δγ = 0
    end

    # 2. Unit normal and consistency parameter
    n  = s_tr / norm_s_tr
    Δγ = f_tr / (2μ * (one(T) + K′ / (3μ)))

    # 3. Update equivalent plastic strain and stress (np1 = n+1)
    α_np1 = α_n + sqrt_2_3 * Δγ
    s_np1 = s_tr - 2μ * Δγ * n

    return s_np1, typeof(st)(α_np1)     # typeof keeps the stored state type
end