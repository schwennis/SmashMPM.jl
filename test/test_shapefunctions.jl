# ---------------------------------------------------------------------------- #
#                 Shape functions and transfer bases (APIC, PolyPIC)           #
# ---------------------------------------------------------------------------- #
# The conservation properties of the whole solver rest on a few identities of
# the stencil. They are checked here directly, at random particle positions:
#   Σ w_I = 1                              mass / momentum conservation
#   Σ w_I (x_I - x_p) = 0                  linear completeness
#   Σ w_I (x_I - x_p)(x_I - x_p)ᵀ = Δx²/4 I   APIC inertia = M_inv⁻¹ (angular momentum)
#   PolyPIC: the 10 modes are orthogonal with the stated norms (exact projection)

@testset "Quadratic B-spline (1D)" begin
    w  = S.quadspline_1d
    xs = range(-2, 2; length = 801)
    @test (w(0.0), w(0.5), w(1.0), w(1.5), w(3.0)) == (0.75, 0.5, 0.125, 0.0, 0.0)
    @test all(x -> w(x) == w(-x) && w(x) >= 0, xs)

    # C¹ at the knots ½ and 3/2: value and slope have no jump
    h = 1e-6
    @test maximum(k -> abs(w(k + h) - w(k - h)), (0.5, 1.5)) < 1e-5
    dw(x) = (w(x + 1e-8) - w(x - 1e-8)) / 2e-8
    @test maximum(k -> abs(dw(k + h) - dw(k - h)), (0.5, 1.5)) < 1e-4
end

@testset "Stencil identities at random positions" begin
    spline = QuadraticSpline()
    rng    = Xoshiro(42)
    dx     = 0.37
    inv_dx = 1 / dx
    origin = SVector(-1.3, 0.2, 5.0)

    e_pu = e_lin = e_inertia = e_table = 0.0
    for _ in 1:200
        x    = origin + dx * SVector{3}(3 .+ 10 .* rand(rng, 3))
        g    = S.grid_position(x, inv_dx, origin)
        base = S.get_support_base(spline, g)
        W    = S.stencil_weights(spline, g, base)

        Σw, Σwr, D = 0.0, zero(V3), zero(M3)
        for di in 0:2, dj in 0:2, dk in 0:2
            node = SVector(base[1] + di, base[2] + dj, base[3] + dk)
            N    = shapefunction(spline, g - node)
            r    = S.node_position(origin, inv_dx, node...) - x
            Σw  += N
            Σwr += N * r
            D   += N * r * r'
            e_table = max(e_table, abs(S.stencil_weight(W, di, dj, dk) - N))
        end
        e_pu      = max(e_pu, abs(Σw - 1))
        e_lin     = max(e_lin, norm(Σwr) / dx)
        e_inertia = max(e_inertia, maximum(abs, D * S.M_inv(spline, inv_dx) - I3))
    end
    @test e_pu      < 1e-13       # partition of unity
    @test e_lin     < 1e-13       # linear completeness
    @test e_inertia < 1e-12       # D_p = Δx²/4 I, consistent with M_inv
    @test e_table   < 1e-14       # tabulated 1D weights = node-wise evaluation

    # node_position is the inverse of grid_position
    @test S.grid_position(S.node_position(origin, inv_dx, 7, -2, 13), inv_dx, origin) ≈ SVector(7, -2, 13)

    # a shape function without its own table uses the generic node-wise fallback
    struct _NodewiseSpline <: S.AbstractShapeFunction end
    @test S.stencil_weights(_NodewiseSpline(), SVector(1.0, 2.0, 3.0), (0, 1, 2)) isa S.NodewiseWeights
end


# ---------------------------------------------------------------------------- #
#                               PolyPIC basis                                  #
# ---------------------------------------------------------------------------- #
# Weights and the 10 basis values of all 27 nodes of a particle at grid position g
function poly_stencil(g::SVector{3, T}) where {T}
    spline = QuadraticSpline()
    hs     = S.higher_stencil(PolyPIC(), spline, g)
    base   = S.get_support_base(spline, g)
    nodes  = map(Iterators.product(0:2, 0:2, 0:2)) do (di, dj, dk)
        ρ = -(g - SVector(base[1] + di, base[2] + dj, base[3] + dk))
        (N = shapefunction(spline, -ρ), ρ = ρ, φ = S.higher_basis(hs, ρ, (di, dj, dk)))
    end
    return hs, vec(nodes)
end

"G2P exactly as in the kernel: (v, C Δx, coefficients of the higher modes, extracted energy, damping)"
function poly_g2p(nodes, v_I, hs_old, hs_new)
    T = eltype(first(v_I))
    vel, CΔx = zero(SVector{3, T}), zero(SMatrix{3, 3, T, 9})
    b = S.zero_higher(PolyPIC(), T)
    for (n, v) in zip(nodes, v_I)
        vel += n.N * v
        CΔx += 4 * n.N * v * n.ρ'                   # B M⁻¹ in units of Δx
        b    = S.accumulate_higher(b, n.N, v, n.φ)
    end
    c, e, d = S.finalize_higher(PolyPIC(), b, hs_old, hs_new)
    return vel, CΔx, c, e, d
end

@testset "PolyPIC: orthogonal basis, exact projection, energy-safe damping" begin
    rng = Xoshiro(1)

    # Gram matrix Σ w φ φᵀ = diag(1, ¼, ¼, ¼, 1/16, 1/16, 1/16, ν₂(t_x), ν₂(t_y), ν₂(t_z))
    e_gram = 0.0
    for _ in 1:50
        g = SVector{3}(10 .+ 3 .* rand(rng, 3))
        hs, nodes = poly_stencil(g)
        t = S.node_offset(g)
        Gram = sum(n.N * vcat(1.0, n.ρ, n.φ) * vcat(1.0, n.ρ, n.φ)' for n in nodes)
        ν2   = (1 .- 4 .* t .^ 2) .^ 2 .* (3 .- 4 .* t .^ 2) ./ 16
        e_gram = max(e_gram, maximum(abs, Gram - Diagonal(vcat(1.0, fill(0.25, 3), fill(1 / 16, 3), ν2))))
    end
    @test e_gram < 1e-13

    # A grid field inside the span of the 10 modes is recovered exactly, and its
    # kinetic energy equals the energy of the modes (Parseval)
    g = SVector(7.3, 4.81, 5.07)
    hs, nodes = poly_stencil(g)
    c0, Cl, ch = SVector{3}(randn(rng, 3)), SMatrix{3, 3}(randn(rng, 3, 3)), SMatrix{3, 6}(randn(rng, 3, 6))
    v_I = [c0 + Cl * n.ρ + ch * n.φ for n in nodes]
    vel, CΔx, c, e_high, e_damp = poly_g2p(nodes, v_I, hs, hs)
    @test vel ≈ c0 && CΔx ≈ Cl && c ≈ ch
    @test e_damp == 0                                         # particle at rest: no damping
    E_grid = 0.5 * sum(n.N * dot(v, v) for (n, v) in zip(nodes, v_I))
    @test E_grid ≈ 0.5 * (dot(c0, c0) + 0.25 * sum(abs2, Cl)) + e_high

    # Moving particle: the G modes put at most the extracted energy back onto the
    # grid (never more), the difference is reported as damping loss
    worst, e_bilinear = 0.0, 0.0
    for _ in 1:30
        g_new  = g .+ 0.6 .* (SVector{3}(rand(rng, 3)) .- 0.5)
        hs_new = S.higher_stencil(PolyPIC(), QuadraticSpline(), g_new)
        _, _, c2, _, d2 = poly_g2p(nodes, v_I, hs, hs_new)
        e_in  = sum(k -> 0.5 * hs.ν2[k]     * sum(abs2, ch[:, 3 + k]), 1:3)
        e_out = sum(k -> 0.5 * hs_new.ν2[k] * sum(abs2, c2[:, 3 + k]), 1:3)
        worst = max(worst, abs(e_out + d2 - e_in) / e_in)     # energy balance of the G modes
        e_bilinear = max(e_bilinear, relerr(c2[:, 1:3], ch[:, 1:3]))
    end
    @test worst < 1e-12
    @test e_bilinear < 1e-12                                  # bilinear modes are never damped

    # Float32 next to a cell face (t → ½): the factored G stays accurate
    e32 = 0.0f0
    for s in (1f-2, 1f-3, 1f-4), axis in 1:3
        g32 = setindex(SVector{3, Float32}(7.3, 4.81, 5.07), 10.5f0 - s, axis)
        hs32, nodes32 = poly_stencil(g32)
        v32 = [c0 + Cl * n.ρ + ch * n.φ for n in nodes32]
        v32 = SVector{3, Float32}.(v32)
        _, _, c32, _, _ = poly_g2p(nodes32, v32, hs32, hs32)
        e32 = max(e32, maximum(abs, c32[:, 3 + axis] - ch[:, 3 + axis]) / maximum(abs, ch))
    end
    @test e32 < 1f-3

    @test_throws ErrorException S.check_transfer(PolyPIC(), nothing)   # PolyPIC needs the quadratic spline
end
