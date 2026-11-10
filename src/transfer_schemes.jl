abstract type AbstractTransferScheme end

"""APIC (MLS-MPM). Equivalent to AFLIP with α ≡ 0 at compile time."""
struct APIC <: AbstractTransferScheme end

"""Affine-augmented FLIP with angular-momentum-conserving C update."""
@kwdef struct AFLIP{T} <: AbstractTransferScheme
    α::T = 0.99
    function AFLIP(α::T) where T
        @assert (0 <= α <= 1) "AFLIP: α must be in [0, 1]"
        return new{T}(α)
    end
end


# Kernel helpers (resolved at compile time)
@inline uses_flip(::APIC)  = false
@inline uses_flip(::AFLIP) = true

@inline flip_ratio(::APIC, ::Type{T}) where {T}   = zero(T)
@inline flip_ratio(s::AFLIP, ::Type{T}) where {T} = T(s.α)

# β_p of the position correction: always 0 for AFLIP, placeholder for ASFLIP
@inline advection_ratio(::AbstractTransferScheme, ::Type{T}, args...) where {T} = zero(T)