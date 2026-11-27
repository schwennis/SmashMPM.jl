# ---------------------------------------------------------------------------- #
#                         Particle <-> grid transfer schemes                   #
# ---------------------------------------------------------------------------- #
#   APIC()      modes 1, x, y, z                      (affine, 4 modes)
#   PolyPIC()   modes 1, x, y, z, xy, xz, yz, Gx, Gy, Gz   (2nd order, 10 modes)
#
# Both take the keyword `heating = true` to book the kinetic energy that the
# transfer loses as heat into the internal energy of the particles (see the
# section "Heat bookkeeping" below). The flag is a type parameter, so all
# heating code is removed at compile time when it is switched off.
#
# Notation (everything per axis, lengths in units of Δx):
#   t = x_p - x_c ∈ [-½, ½)      offset of the particle from the nearest node
#   ρ = x_I - x_p                 offset of node I from the particle
#   w(ρ)                          quadratic B-spline weight
#
# Per-axis basis, orthogonal with respect to the weights of the CURRENT stencil:
#   1,   ρ,   G(ρ) = ρ² - aρ - ¼,   a = t(1 - 4t²)
# Norms Σ w φ²:
#   ν₀ = 1,   ν₁ = ¼,   ν₂(t) = (1 - 4t²)²(3 - 4t²)/16       (ν₂ ≤ 3/16)
# The 3D modes are tensor products, their norms are products of the per-axis
# norms: ν(xy) = 1/16, ν(G_x) = ν₂(t_x) (the other two factors are ν₀ = 1).
#
# The G modes are rebuilt with the a of the NEW stencil in the P2G. Only then
# are all higher modes orthogonal to 1 and ρ at the new stencil, i.e. they
# carry neither momentum nor angular momentum and L stays exact.
# ---------------------------------------------------------------------------- #
abstract type AbstractTransfer end

"""
    APIC(; heating=false)

Affine particle-in-cell transfer (modes 1, x, y, z). With `heating = true` the
kinetic energy lost in the transfer is added to the internal energy of the
particles.
"""
struct APIC{H} <: AbstractTransfer end
APIC(; heating::Bool=false) = APIC{heating}()

"""
    PolyPIC(; heating=false)

Second-order PolyPIC transfer with 10 modes (1, x, y, z, xy, xz, yz, Gx, Gy, Gz).
Only implemented for the `QuadraticSpline`. The G modes are regularized with the
parameter-free minimal rule (energy neutral, see `finalize_higher`).
With `heating = true` the transfer losses are booked as heat.
"""
struct PolyPIC{H} <: AbstractTransfer end
PolyPIC(; heating::Bool=false) = PolyPIC{heating}()

@inline transfer_heating(::APIC{H}) where {H}    = H
@inline transfer_heating(::PolyPIC{H}) where {H} = H

"""Host-side check that a transfer scheme can be used with a shape function."""
check_transfer(::APIC, spline) = nothing
check_transfer(::PolyPIC, ::QuadraticSpline) = nothing
check_transfer(tr::PolyPIC, spline) =
    error("$(typeof(tr)) is only implemented for the QuadraticSpline, not for $(typeof(spline))")


# ---------------------------------------------------------------------------- #
#                         Per-axis basis of the quadratic spline               #
# ---------------------------------------------------------------------------- #
"""
    PolyStencil

Per-axis data of the G modes at one particle position:
- `G[k, d]`: value of G along axis k at the stencil node d = 1, 2, 3
  (offsets ρ = -1-t, -t, 1-t, i.e. di = d - 1 in the kernel loop)
- `ν2[k]`:   norm of G along axis k (units of Δx⁴)
"""
struct PolyStencil{T}
    G::SMatrix{3, 3, T, 9}
    ν2::SVector{3, T}
end

"""Offset t ∈ [-½, ½) of the particle from its nearest node (per axis, units of Δx)."""
@inline function node_offset(grid_pos::SVector{3, T}) where {T}
    # Same nodes as get_support_base: center node = floor(g - ½) + 1.
    # Float floor (no Int conversion), so NaN positions do not throw here.
    s = grid_pos .- T(0.5)
    return s .- floor.(s) .- T(0.5)
end

@inline poly_a(t)  = t * (1 - 4t^2)
@inline poly_ν2(t) = (1 - 4t^2)^2 * (3 - 4t^2) / 16

# G at the three nodes of one axis, in factored form. ρ² - aρ - ¼ directly would
# cancel: near t = ±½ G is O(s²) at the two heavy nodes (s = ½ - |t|) while the
# three terms are O(1). With u = 1 - 2t, v = 1 + 2t (both exact in floating point
# for |t| ≤ ½) every value below has full relative precision:
#   G(-1-t) =  ¼ v² (3 - 4t²),   G(-t) = -¼ (uv)²,   G(1-t) = ¼ u² (3 - 4t²)
#   ν₂      = (uv)² (3 - 4t²) / 16
@inline function _poly_axis(t::T) where {T}
    u, v = 1 - 2t, 1 + 2t
    q    = 3 - 4t^2
    uv2  = (u * v)^2
    return SVector{3, T}(v^2 * q / 4, -uv2 / 4, u^2 * q / 4), uv2 * q / 16
end

# APIC needs no extra stencil data. `nothing` everywhere below makes the higher
# mode code disappear at compile time.
@inline higher_stencil(::APIC, spline, grid_pos) = nothing
@inline function higher_stencil(::PolyPIC, ::QuadraticSpline, grid_pos::SVector{3, T}) where {T}
    t = node_offset(grid_pos)
    Gx, νx = _poly_axis(t[1])
    Gy, νy = _poly_axis(t[2])
    Gz, νz = _poly_axis(t[3])
    G = SMatrix{3, 3, T, 9}(Gx[1], Gy[1], Gz[1], Gx[2], Gy[2], Gz[2], Gx[3], Gy[3], Gz[3])
    return PolyStencil{T}(G, SVector{3, T}(νx, νy, νz))
end

"""
    higher_basis(stencil, ρ, (di, dj, dk)) -> SVector{6}

Values of the higher modes (xy, xz, yz, Gx, Gy, Gz) at the stencil node with
offset `ρ = x_I - x_p` (units of Δx) and loop offsets di, dj, dk ∈ 0:2.
"""
@inline higher_basis(::Nothing, ρ, d) = nothing
@inline function higher_basis(s::PolyStencil{T}, ρ::SVector{3, T}, (di, dj, dk)) where {T}
    return SVector{6, T}(ρ[1] * ρ[2], ρ[1] * ρ[3], ρ[2] * ρ[3],
                         s.G[1, di + 1], s.G[2, dj + 1], s.G[3, dk + 1])
end


# ---------------------------------------------------------------------------- #
#                         G2P: projection onto the higher modes                #
# ---------------------------------------------------------------------------- #
# The accumulator is the (unnormalized) projection b_α = Σ_I w_I φ_α(I) v_I,
# one column per mode, stored as a 3x6 matrix.
@inline zero_higher(::APIC, ::Type{T}) where {T}    = nothing
@inline zero_higher(::PolyPIC, ::Type{T}) where {T} = zero(SMatrix{3, 6, T, 18})

@inline accumulate_higher(::Nothing, N, v, φ) = nothing
@inline accumulate_higher(b::SMatrix{3, 6}, N, v, φ) = b + (N * v) * φ'

@inline _col(M::SMatrix{3, 6, T}, n) where {T} = SVector{3, T}(M[1, n], M[2, n], M[3, n])

"""
    finalize_higher(transfer, b, stencil_old, stencil_new) -> (c, e_extracted, e_damping)

Divides the projections by the norms of the OLD stencil, c_α = b_α / ν_α, and
applies the exact minimal regularization to the G modes:

    c_Gk ← c_Gk · min(1, √(ν₂(t_k) / ν₂(t'_k)))

t'_k is the offset at the new position x + Δt·c₀₀₀ (= wrap(t_k + c₀₀₀,k Δt/Δx)).
In the P2G the mode puts ν₂(t') |c|² back onto the grid. With the factor this is
at most ν₂(t)|c|², which is what the G2P extracted: the transfer of the G mode
never creates energy. Without motion the factor is 1 (no damping at all).
The norms of the bilinear modes are constant (1/16), they need no damping.

Returns the regularized coefficients (3x6, columns xy, xz, yz, Gx, Gy, Gz) and,
per unit mass, the kinetic energy ½ Σ ν_α |c_α|² carried by the higher modes and
the energy that the P2G cannot put back (½ (ν₂ - ν₂')₊ |c_G|², only if ν₂' < ν₂).
"""
@inline finalize_higher(::APIC, b, old, new) = (nothing, false, false)  # false: type neutral 0

@inline function finalize_higher(::PolyPIC, b::SMatrix{3, 6, T, 18},
                                 old::PolyStencil{T}, new::PolyStencil{T}) where {T}
    half  = T(0.5)
    ν_mix = T(1 / 16)
    # G and ν₂ are evaluated without cancellation (see _poly_axis), so c = b/ν stays
    # accurate up to a few eps even for t → ±½ (b and ν are both O(s²) there).
    # The guard only excludes ν = 0 (t = -½ exactly: only 2 nodes, b = 0) and
    # subnormal norms. Not extracting the mode is dissipative and Bessel safe.
    ν_min = sqrt(floatmin(T))

    c_xy = _col(b, 1) * T(16)
    c_xz = _col(b, 2) * T(16)
    c_yz = _col(b, 3) * T(16)
    e    = half * ν_mix * (dot(c_xy, c_xy) + dot(c_xz, c_xz) + dot(c_yz, c_yz))

    # Written out per axis (no ntuple closure / hcat): this has to stay allocation
    # free and GPU compilable.
    gx, ex, dx_ = _g_mode(_col(b, 4), old.ν2[1], new.ν2[1], ν_min)
    gy, ey, dy_ = _g_mode(_col(b, 5), old.ν2[2], new.ν2[2], ν_min)
    gz, ez, dz_ = _g_mode(_col(b, 6), old.ν2[3], new.ν2[3], ν_min)

    c = SMatrix{3, 6, T, 18}(c_xy[1], c_xy[2], c_xy[3], c_xz[1], c_xz[2], c_xz[3],
                             c_yz[1], c_yz[2], c_yz[3], gx[1], gx[2], gx[3],
                             gy[1], gy[2], gy[3], gz[1], gz[2], gz[3])
    return c, e + ex + ey + ez, dx_ + dy_ + dz_
end

"""One G mode: c = b/ν (old stencil), minimal damping factor, extracted energy, damping loss."""
@inline function _g_mode(b::SVector{3, T}, ν_o::T, ν_n::T, ν_min::T) where {T}
    half = T(0.5)
    c    = ν_o > ν_min ? b / ν_o : zero(SVector{3, T})
    c2   = dot(c, c)
    f    = ν_n > ν_o ? sqrt(ν_o / ν_n) : one(T)                  # min(1, √(ν₂/ν₂'))
    return f * c, half * ν_o * c2, half * max(ν_o - ν_n, zero(T)) * c2
end


# ---------------------------------------------------------------------------- #
#                         P2G: velocity of the higher modes                    #
# ---------------------------------------------------------------------------- #
@inline eval_higher(::Nothing, φ, ::Type{T}) where {T} = zero(SVector{3, T})
@inline eval_higher(c::SMatrix{3, 6, T, 18}, φ::SVector{6, T}, ::Type{T}) where {T} = c * φ


# ---------------------------------------------------------------------------- #
#                               Heat bookkeeping                               #
# ---------------------------------------------------------------------------- #
# The kinetic energy of a particle is the energy of its velocity field,
#   K_p = ½ m_p Σ_α ν_α |c_α|²   (APIC: ½ m_p (|v|² + ¼ Δx² ‖C‖²)).
# One transfer cycle loses kinetic energy in three places. All three are ≥ 0:
#
#   1. G2P projection (Bessel):  ½ m_p (Σ_I w_I |v_I|² - Σ_α ν_α |c_α|²)
#      the part of the grid field that the modes of the particle cannot carry
#   2. G-mode regularization:    ½ m_p Σ_k (ν₂ - ν₂')₊ |c_Gk|²   (PolyPIC only)
#   3. P2G mixing (Jensen):      per node  ΔE_I = Σ_p ½ m_p w_pI |v_pI|² - ½ |p_I|² / m_I
#      different particles (and bodies in contact) put different velocities
#      onto the same node, the node only keeps their mass-weighted mean
#
# 1 and 2 are known per particle inside the G2P2G kernel. 3 is only known after
# all particles have scattered: the P2G accumulates Σ_p ½ m_p w |v_pI|² in the
# grid field e_mix, `mixing_loss!` subtracts the node kinetic energy (before the
# kick) and the next G2P hands ΔE_I back to the particles in proportion to their
# mass share w_pI m_p / m_I. The G2P stencil of a particle is its last P2G stencil,
# so Σ_p w_pI m_p = m_I and the whole ΔE_I is booked (one step later).
#
# With heating, kinetic + internal + pending (Σ_I e_mix) energy is therefore
# conserved by the transfer itself; only forces, boundary conditions and the
# time integration change it (see `energy_budget`).
