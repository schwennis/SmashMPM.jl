abstract type AbstractTransferScheme end

# All schemes carry H::Bool as a type parameter: `heat = true` books the kinetic energy
# that the particle-grid transfer dissipates into the specific internal energy e:
#   * P2G loss of every node, L_i = Σ_p N ½ m |v_p + C_p r|² - ½ |p_i|²/m_i ≥ 0 (Jensen),
#     shared by the particles of the node by mass (like the KE fix-up of ALE remaps),
#   * G2P residual of the affine fit per particle, ≥ 0,
#   * for AFLIP the blend loss ½ m α (1-α) ‖ū - u‖² ≥ 0, the other terms times (1-α).
# Every booked term is ≥ 0 and spread over the nodes, so no particle gets negative or
# concentrated heat. Not booked: the (sign-indefinite) energy error of the FLIP path
# and the kinetic energy change of the spin-fix. Mass, momentum and angular momentum
# are not affected (only e changes). Switch: AFLIP(α; heat = true), APIC(heat = true).

"""
    APIC(; heat = false)

APIC (MLS-MPM). Equivalent to AFLIP with α ≡ 0 at compile time.
"""
struct APIC{H} <: AbstractTransferScheme end
APIC(; heat::Bool = false) = APIC{heat}()

"""
    AFLIP(α = 0.99; heat = false)

Affine-augmented FLIP with angular-momentum-conserving C update (C-FLIP + spin-fix).
"""
struct AFLIP{T, H} <: AbstractTransferScheme
    α::T
    function AFLIP{T, H}(α::T) where {T, H}
        @assert (0 <= α <= 1) "AFLIP: α must be in [0, 1]"
        return new{T, H}(α)
    end
end
AFLIP(α::T = 0.99; heat::Bool = false) where {T <: Real} = AFLIP{T, heat}(α)

# Kernel helpers (resolved at compile time)
@inline uses_flip(::APIC)  = false
@inline uses_flip(::AFLIP) = true

@inline books_heat(::APIC{H})  where {H}    = H
@inline books_heat(::AFLIP{T, H}) where {T, H} = H

# v̄ⁿ, C̄ⁿ (G2P of pⁿ/m) are needed for FLIP and the heat bookkeeping
@inline needs_bar_n(s::AbstractTransferScheme) = uses_flip(s) || books_heat(s)

@inline flip_ratio(::AbstractTransferScheme, ::Type{T}) where {T} = zero(T)
@inline flip_ratio(s::AFLIP, ::Type{T}) where {T} = T(s.α)

# β_p of the position correction: always 0 for AFLIP, placeholder for ASFLIP
@inline advection_ratio(::AbstractTransferScheme, ::Type{T}, args...) where {T} = zero(T)
