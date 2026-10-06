# ---------------------------------------------------------------------------- #
#                          Murnaghan Equation of State                         #
# ---------------------------------------------------------------------------- #
@kwdef struct MurnaghanEOS{T} <: AbstractEquationOfState
    ρ::T       # Reference density
    K0::T       # Bulk modulus at reference density
    n::T        # Murnaghan-Exponent
    η_limit::T  # below this relative density p = 0
end

struct MurnaghanState{T} <: AbstractEoSState
    p::T
end


init_state(::MurnaghanEOS{T}) where {T} = MurnaghanState(zero(T))
reference_soundspeed(eos::MurnaghanEOS) = sqrt(eos.K0 / eos.ρ)


# ---------------------------------------------------------------------------- #
#                           update_eos implementation                          #
# ---------------------------------------------------------------------------- #
@inline function update_eos(eos::MurnaghanEOS{T}, ::MurnaghanState{T}, ρ::T, e::T) where {T}
    η = max(ρ / eos.ρ, eps(T))

    if η > eos.η_limit
        @fastmath ηn = η^eos.n
        p  = eos.K0 / eos.n * (ηn - one(T))
        c  = sqrt(max(eos.K0 / eos.ρ * ηn / η, zero(T)))           # η^(n-1)
    else
        p  = zero(T)
        c  = sqrt(max(eos.K0 / eos.ρ * eos.η_limit^(eos.n - one(T)), zero(T)))
    end

    return MurnaghanState(p), c
end