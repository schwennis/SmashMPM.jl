# ---------------------------------------------------------------------------- #
#                     Transfer energy bookkeeping (host side)                  #
# ---------------------------------------------------------------------------- #
# See the section "Heat bookkeeping" in src/transfer.jl.

"""
Mixing loss of one node relative to the reference velocity u (read velocity of
the node): ΔE = Σ_p ½ m_p w |v_pI - u|² - ½ m |v̄ - u|², v̄ = p/m. This equals
Σ_p ½ m_p w |v_pI|² - ½ |p|²/m for every u, but does not cancel at high speed.
"""
@inline function _node_mixing_loss(e_dep::T, m::T, px::T, py::T, pz::T,
                                   m_r::T, prx::T, pry::T, prz::T) where {T}
    cut = sqrt(floatmin(T))
    m > cut || return zero(T)
    u  = m_r > cut ? SVector{3, T}(prx, pry, prz) / m_r : zero(SVector{3, T})
    dv = SVector{3, T}(px, py, pz) / m - u
    return e_dep - T(0.5) * m * dot(dv, dv)
end

"""
    mixing_loss!(transfer, grid)

Converts the deposited particle energy in `grid.state_write.e_mix` into the P2G
mixing loss of every node, ΔE_I = Σ_p ½ m_p w |v_pI - u_I|² - ½ m_I |v̄_I - u_I|²
with the reference velocity u_I of the node in the READ state (the same one the
P2G used, see `reference_velocity`).
Must be called right after the P2G, before forces are kicked into the momentum
and before the grid swap. Does nothing without heating.
"""
mixing_loss!(transfer::AbstractTransfer, grid::AbstractGrid) =
    transfer_heating(transfer) ? _mixing_loss!(grid) : nothing

function _mixing_loss!(grid::DenseGrid)
    s, r = grid.state_write, grid.state_read
    s.e_mix .= _node_mixing_loss.(s.e_mix, s.mass, s.momentum.x, s.momentum.y, s.momentum.z,
                                  r.mass, r.momentum.x, r.momentum.y, r.momentum.z)
    return nothing
end

function _mixing_loss!(grid::SparseGrid)
    n = grid.n_active_write
    n == 0 && return nothing
    s, r = grid.state_write, grid.state_read
    backend = KernelAbstractions.get_backend(s.mass)
    sparse_mixing_loss_kernel!(backend)(
        s.e_mix, s.mass, s.momentum.x, s.momentum.y, s.momentum.z,
        grid.map_write.tile_keys, grid.map_read,
        r.mass, r.momentum.x, r.momentum.y, r.momentum.z; ndrange = (TILE_VOLUME, n))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# One thread per node of the active write tiles; the read tile with the same key
# holds the reference velocity (tile not active in the read state: u = 0)
@kernel function sparse_mixing_loss_kernel!(e, m, px, py, pz, keys_write, map_read, m_r, prx, pry, prz)
    l, t = @index(Global, NTuple)
    T  = eltype(e)
    tr = find_tile(map_read, keys_write[t])
    if tr == 0
        e[l, t] = _node_mixing_loss(e[l, t], m[l, t], px[l, t], py[l, t], pz[l, t],
                                    zero(T), zero(T), zero(T), zero(T))
    else
        e[l, t] = _node_mixing_loss(e[l, t], m[l, t], px[l, t], py[l, t], pz[l, t],
                                    m_r[l, tr], prx[l, tr], pry[l, tr], prz[l, tr])
    end
end


# ---------------------------------------------------------------------------- #
#                                 Energy budget                                #
# ---------------------------------------------------------------------------- #
@inline function _node_kinetic(m::T, px::T, py::T, pz::T) where {T}
    m > sqrt(floatmin(T)) || return zero(T)
    return T(0.5) * (px * px + py * py + pz * pz) / m
end

# Read state of the grid (= state after the last step incl. kick and BCs)
_read_fields(grid::DenseGrid) = grid.state_read
function _read_fields(grid::SparseGrid)
    s = grid.state_read
    n = grid.n_active_read
    v(f) = view(f, :, 1:n)
    return (mass = v(s.mass), momentum = (x = v(s.momentum.x), y = v(s.momentum.y), z = v(s.momentum.z)),
            e_mix = v(s.e_mix))
end

"""
    energy_budget(model) -> (kinetic, internal, elastic, pending, total)

- `kinetic`:  kinetic energy of the grid, Σ_I ½ |p_I|² / m_I (read state)
- `internal`: Σ_p m_p e_p of all materials with an internal energy
              (SolidMaterial: specific internal energy e, incl. the volume work)
- `elastic`:  Σ_p V0_p W_p, stored elastic energy (`strain_energy`; SolidMaterial:
              shear part only, the volumetric part is in e)
- `pending`:  mixing loss of the last P2G that the next G2P books as heat
              (only non-zero with `heating = true`)
- `total`:    sum of all four

With `heating = true` the transfer itself conserves `total` up to rounding; the
remaining change comes from the time integration (leapfrog fluctuation, mainly at
contacts), the artificial viscosity and the boundary conditions.
Diagnostic only (one reduction per field), not meant for every time step.
"""
function energy_budget(model::MPMModel{T}) where {T}
    g = _read_fields(model.grid)
    kinetic = sum(_node_kinetic.(g.mass, g.momentum.x, g.momentum.y, g.momentum.z); init=zero(T))
    pending = sum(g.e_mix; init=zero(T))

    internal = zero(T)
    elastic  = zero(T)
    for ps in model.particle_sets
        mat = ps.material
        p   = ps.particles
        if has_internal_energy(mat)
            internal += sum(p.mass .* internal_energy.(Ref(mat), p.mat_state); init=zero(T))
        end
        # AoSoA padding particles have V0 = 0 (their F = I, no contribution)
        elastic += sum(p.V0 .* strain_energy.(Ref(mat), p.mat_state, p.F); init=zero(T))
    end
    return (kinetic = kinetic, internal = internal, elastic = elastic, pending = pending,
            total = kinetic + internal + elastic + pending)
end
