@kwdef struct BulkViscosity{T} <: AbstractArtificialViscosity
    c_quad::T = 1.5    # quadratic coefficient (shock spreading)
    c_lin::T  = 0.06   # linear coefficient (damps ringing)
end

@inline function viscous_pressure(av::BulkViscosity, kin::Kinematics{T}, c) where {T}
    divv = tr(kin.C)
    divv >= zero(T) && return zero(T)          # only active in compression
    ρ = kin.m / (det(kin.F) * kin.V0)
    l = kin.dx
    return ρ * l * (av.c_quad * l * divv^2 - av.c_lin * c * divv)
end