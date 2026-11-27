# ---------------------------------------------------------------------------- #
#                         Shared helpers of the test suite                     #
# ---------------------------------------------------------------------------- #
const S  = SmashMPM
const V3 = SVector{3, Float64}
const M3 = SMatrix{3, 3, Float64, 9}
const I3 = one(M3)

"Largest relative deviation of `a` from `b` (max norm, safe for b ≈ 0)."
relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), eps(Float64))

"Allocated bytes of `f(args...)` after a warm-up call (inside a function, so no boxing of the result)."
function allocs(f, args...)
    f(args...)
    return @allocated f(args...)
end

"Run `n` time steps."
run!(model, n) = (for _ in 1:n; step!(model); end; model)

"Stand-in for a particle set that only carries positions (enough for `activate_tiles!`)."
position_set(pts::AbstractVector{<:SVector{3}}) =
    (particles = (pos = StructArray{eltype(pts)}((x = getindex.(pts, 1), y = getindex.(pts, 2),
                                                 z = getindex.(pts, 3))),),)


"""
    box_grid(G, n)

n³ grid (DenseGrid, or SparseGrid(BoxDomain) with every tile of the box active)
with unit mass on every node of the write state, padding 2.
"""
function box_grid(::Type{G}, n) where {G}
    N, dx, o = SVector(n, n, n), 0.5, V3(-1.0, 0.5, 2.0)
    G === DenseGrid && return (g = DenseGrid(dx, N, o, 2); g.state_write.mass .= 1; g)
    g = SparseGrid(dx, N, o, 2)
    nodes = vec([S.node_position(o, 1 / dx, Tuple(I)...) for I in CartesianIndices((n, n, n))])
    activate_tiles!(g, (position_set(nodes),), QuadraticSpline())
    g.state_write.mass .= 1
    return g
end


# ---------------------------------------------------------------------------- #
#                              Grid diagnostics                                #
# ---------------------------------------------------------------------------- #
"""
    foreach_node(f, grid)

Calls `f(x, m, p)` for every node of the READ state that carries mass
(x position, m mass, p momentum). Works for both grid types without converting
a SparseGrid to a dense box.
"""
function foreach_node(f, grid::DenseGrid)
    st = grid.state_read
    for I in CartesianIndices(st.mass)
        m = st.mass[I]
        m > 0 || continue
        f(S.node_position(grid.origin, grid.inv_dx, Tuple(I)...), m, st.momentum[I])
    end
    return nothing
end

function foreach_node(f, grid::SparseGrid)
    st = grid.state_read
    for t in 1:grid.n_active_read, l in 1:S.TILE_VOLUME
        m = st.mass[l, t]
        m > 0 || continue
        i, j, k = S.node_index(grid.map_read.tile_keys[t], l)
        f(S.node_position(grid.origin, grid.inv_dx, i, j, k), m, st.momentum[l, t])
    end
    return nothing
end

"Total mass, momentum and angular momentum (about the origin) of the grid, in Float64."
function grid_totals(grid)
    acc = Ref((M = 0.0, P = zero(V3), L = zero(V3)))
    foreach_node(grid) do x, m, p
        x, p = V3(x), V3(p)
        a = acc[]
        acc[] = (M = a.M + m, P = a.P + p, L = a.L + cross(x, p))
    end
    return acc[]
end

"Mass-weighted center of all particles."
function center_of_mass(model)
    M, X = 0.0, zero(V3)
    for ps in model.particle_sets, (m, x) in zip(ps.particles.mass, ps.particles.pos)
        M += m
        X += m * V3(x)
    end
    return X / M
end

"Particle positions of all sets, concatenated."
all_positions(model) = reduce(vcat, [V3.(collect(ps.particles.pos)) for ps in model.particle_sets])

"""
    set_grid_velocity!(grid::DenseGrid, v)

Overwrites the momentum of the read state with m_I v(x_I). APIC/PolyPIC
reproduce affine fields exactly, so the next G2P hands v(x_p) to the particles.
"""
function set_grid_velocity!(grid::DenseGrid, v)
    st = grid.state_read
    for I in CartesianIndices(st.mass)
        x = S.node_position(grid.origin, grid.inv_dx, Tuple(I)...)
        st.momentum[I] = st.mass[I] * v(x)
    end
    return grid
end


# ---------------------------------------------------------------------------- #
#                 Test material: pressureless dust with heat                   #
# ---------------------------------------------------------------------------- #
# σ = 0: the grid sees no internal forces, so every change of the kinetic
# energy is caused by the transfer. Colliding dust sticks together (perfectly
# inelastic), which makes it ideal for checking the heat bookkeeping.
struct HeatDust{T} <: SmashMPM.AbstractMaterial
    ρ::T
end
struct HeatDustState{T} <: SmashMPM.AbstractMaterialState
    e::T
end
SmashMPM.initial_material_state(::HeatDust{T}) where {T} = HeatDustState(zero(T))
SmashMPM.material_model(::HeatDust, st::HeatDustState, F, C::SMatrix{3, 3, T, 9}, V0, m, dt, dx) where {T} =
    (zero(SMatrix{3, 3, T, 9}), st)
SmashMPM.soundspeed(::HeatDust, st::HeatDustState) = zero(st.e)
SmashMPM.has_internal_energy(::HeatDust) = true
SmashMPM.internal_energy(::HeatDust, st::HeatDustState) = st.e
SmashMPM.add_heat(::HeatDust, st::HeatDustState, Δe) = HeatDustState(st.e + Δe)


# ---------------------------------------------------------------------------- #
#                                Material zoo                                  #
# ---------------------------------------------------------------------------- #
"Every material model / component combination that the type stability tests cover."
material_zoo(T) = (
    "NoMaterialModel"     => NoMaterialModel(),
    "LinearElastic"       => LinearElastic(ρ = T(1000), E = T(1e6), ν = T(0.3)),
    "NeoHookean"          => NeoHookean(ρ = T(1000), E = T(1e6), ν = T(0.3)),
    "Al hypo, Tillotson"  => Aluminum(T),
    "Al hyper, J2, q"     => Aluminum(T; elasticity = :hyperelastic, viscosity = BulkViscosity{T}(),
                                      plasticity = J2Plasticity(σ_y0 = T(2.76e8), H = T(6.8e8))),
    "Fe Murnaghan, GK"    => Iron(T; eos = :murnaghan, damage = :grady_kipp),
    "Basalt GK, q"        => Basalt(T; viscosity = BulkViscosity{T}()),
    "Nylon J2"            => Nylon(T),
)


# ---------------------------------------------------------------------------- #
#                                   Scenes                                     #
# ---------------------------------------------------------------------------- #
"""
Two spinning elastic blocks of different density and stiffness that collide
(contact through the shared grid). Exercises every term of the transfer:
translation, rotation, deformation, multi-body nodes.
"""
function colliding_blocks(; grid_type = DenseGrid, transfer = APIC(), domain = S.AutoDomain(),
                            T = Float64, boundary_condition = NullBoundaryCondition())
    soft = NeoHookean(ρ = T(1000), E = T(1e6), ν = T(0.3))
    hard = NeoHookean(ρ = T(2000), E = T(4e6), ν = T(0.25))
    a = Body(RectangularPrism{T}(width = T(0.4), height = T(0.4), depth = T(0.4),
                                 center = SVector{3, T}(-0.25, 0.03, 0)),
             SVector{3, T}(2, 0, 0.1), SVector{3, T}(0, 1, 3), soft)
    b = Body(RectangularPrism{T}(width = T(0.3), height = T(0.3), depth = T(0.3),
                                 center = SVector{3, T}(0.25, -0.05, 0.02)),
             SVector{3, T}(-1, 0.2, 0), SVector{3, T}(2, 0, -1), hard)
    setup = SimulationSetup(dx = T(0.1), t_max = T(10), dt_max = T(1e-3), padding = 4,
                            buffer_width = SVector(4, 4, 4), grid_type = grid_type,
                            transfer = transfer, domain = domain, boundary_condition = boundary_condition)
    # heating = true only exercises the heat code path here; NeoHookean has no
    # internal energy to store it, which build_mpm_model rightly warns about
    return with_logger(NullLogger()) do
        build_mpm_model((a, b), setup)
    end
end

"Two colliding blocks of pressureless dust (perfectly inelastic, see HeatDust)."
function dust_collision(; grid_type = DenseGrid, transfer = APIC(), T = Float64)
    mat = HeatDust(T(1000))
    a = RectangularPrism{T}(width = T(0.5), height = T(0.5), depth = T(0.4), center = SVector{3, T}(-0.3, 0.05, 0))
    b = RectangularPrism{T}(width = T(0.4), height = T(0.4), depth = T(0.4), center = SVector{3, T}(0.27, -0.05, 0.02))
    bodies = (Body(a, SVector{3, T}(1, 0.1, 0), SVector{3, T}(0, 0, 2), mat),
              Body(b, SVector{3, T}(-1.5, 0, 0.2), SVector{3, T}(1, 0, 0), mat))
    setup = SimulationSetup(dx = T(0.1), t_max = T(10), padding = 4, buffer_width = SVector(5, 5, 5),
                            dt_max = T(0.01), grid_type = grid_type, transfer = transfer)
    return build_mpm_model(bodies, setup)
end

"Head-on impact of two aluminium cubes at 2 × 300 m/s (Tillotson, hypoelastic)."
function al_impact(; transfer = APIC(), cfl = 0.4, grid_type = SparseGrid, T = Float64)
    a = RectangularPrism{T}(width = T(2e-3), height = T(2e-3), depth = T(2e-3), center = SVector{3, T}(-1.2e-3, 0, 0))
    b = RectangularPrism{T}(width = T(2e-3), height = T(2e-3), depth = T(2e-3), center = SVector{3, T}(1.2e-3, 0, 0))
    bodies = (Body(a, SVector{3, T}(300, 0, 0), zero(SVector{3, T}), Aluminum(T)),
              Body(b, SVector{3, T}(-300, 0, 0), zero(SVector{3, T}), Aluminum(T)))
    setup = SimulationSetup(dx = T(2.5e-4), t_max = T(1), padding = 4, buffer_width = SVector(4, 4, 4),
                            dt_max = T(1e-6), cfl_number = T(cfl), grid_type = grid_type, transfer = transfer)
    return build_mpm_model(bodies, setup)
end

"Single elastic block, optionally under an external force."
function single_block(; grid_type = DenseGrid, v0 = zero(V3), ω = zero(V3), T = Float64,
                        external_force = NoExternalForce(), t_max = 1.0, transfer = APIC(),
                        material = NeoHookean(ρ = T(1000), E = T(1e6), ν = T(0.3)),
                        dx = 0.1, cfl = 0.4, edge = 0.4)
    shape = RectangularPrism{T}(width = T(edge), height = T(edge), depth = T(edge))
    body  = Body(shape, SVector{3, T}(v0), SVector{3, T}(ω), material)
    setup = SimulationSetup(dx = T(dx), t_max = T(t_max), dt_max = T(1), cfl_number = T(cfl),
                            padding = 3, buffer_width = SVector(3, 3, 3), grid_type = grid_type,
                            external_force = external_force, transfer = transfer)
    return build_mpm_model((body,), setup)
end
