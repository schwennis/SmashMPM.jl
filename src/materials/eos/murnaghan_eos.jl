# ---------------------------------------------------------------------------- #
#                          Murnaghan Eqation of State                          #
# ---------------------------------------------------------------------------- #

struct MurnaghanEOS{T} <: AbstractEquationOfState
    ρ0::T   # Reference density
    K0::T   # Bulk modulus at reference density
    n::T    # Murnaghan exponent n
    η_limit::T  # η limit for the EOS
end

struct MurnaghanState{T} <: AbstractEoSState
    p::T    # Pressure
end


function init_eos_state(eos::MurnaghanEOS{T}) where {T}
    # Start with p = 0, e =0
    p0 = zero(T)
    state = MurnaghanState{T}(p0)
    
    # Reference for Murnaghan: c0 = sqrt(K0 / ρ0)
    c0 = sqrt(eos.K0 / eos.ρ0)
    
    return state, c0
end

# ---------------------------------------------------------------------------- #
#                           update_eos implementation                          #
# ---------------------------------------------------------------------------- #
function update_eos(eos::MurnaghanEOS{T}, eos_state::MurnaghanState{T}, ρ::T, stress_work::T, dt::T) where {T}
    eta = ρ / eos.ρ0
    eta_safe = max(eta, eps(T))

    eta_pow_n = eta_safe^eos.n

    if eta_safe > eos.η_limit
        p_new = eos.K0 / eos.n * (eta_pow_n - one(T))
        c = sqrt(max(eos.K0 / eos.ρ0 * eta_pow_n/eta_safe, zero(T)))  # η^(n-1) = η^n / η
    else
        p_new = zero(T)
        c = sqrt(max(eos.K0 / eos.ρ0 * eos.η_limit^(eos.n - one(T)), zero(T)))  # η^(n-1) = η^n / η
    end

    return MurnaghanState{T}(p_new), c
end