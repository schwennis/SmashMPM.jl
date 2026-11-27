abstract type AbstractShapeFunction end

# ---------------------------------------------------------------------------- #
#                                    Helpers                                   #
# ---------------------------------------------------------------------------- #
@inline function grid_position(pos_p::SVector{3, T}, inv_dx::T, origin::SVector{3, T}) where {T}
    return (pos_p - origin) * inv_dx .+ one(T)  # +1 for 1-based indexing
end


# ---------------------------------------------------------------------------- #
#                               Quadratic Spline                               #
# ---------------------------------------------------------------------------- #
struct QuadraticSpline <: AbstractShapeFunction end

@fastmath @inline function shapefunction(::QuadraticSpline, natural_coords::SVector{3, T}) where {T}
    N = quadspline_1d(natural_coords[1]) * quadspline_1d(natural_coords[2]) * quadspline_1d(natural_coords[3])
    
    return N
end

@inline function quadspline_1d(dist_1d::T) where {T}
    abs_d = abs(dist_1d)
    
    w = zero(T)

    if abs_d < T(0.5)
        # Fall 1: |x| < 0.5
        # N(x) = 0.75 - x^2
        w = T(0.75) - abs_d^2
    elseif abs_d < T(1.5)
        # 0.5 <= |x| < 1.5
        # N(x) = 0.5 * (1.5 - |x|)^2
        val = T(1.5) - abs_d
        w = T(0.5) * val^2
    else
        # outside of support
        w = zero(T)
    end

    return w
end


@inline function get_support_base(::QuadraticSpline, grid_coords::SVector{3, T}) where {T}
    i = floor(Int, grid_coords[1] - T(0.5))
    j = floor(Int, grid_coords[2] - T(0.5))
    k = floor(Int, grid_coords[3] - T(0.5))

    return i, j, k
end

@inline get_support_offsets(::QuadraticSpline) = (0:2, 0:2, 0:2)


# ---------------------------------------------------------------------------- #
#                     Precomputed weights of one stencil                       #
# ---------------------------------------------------------------------------- #
# The particle kernels get the weights of a stencil once per particle,
#   W = stencil_weights(spline, grid_coords, base)
# and then the weight of node base + (di, dj, dk) with stencil_weight(W, di, dj, dk).
#
# Generic fallback (every shape function, nothing to implement): the weight is
# evaluated per node with `shapefunction`, exactly as before.
# A tensor-product spline can overload `stencil_weights` with a table of its 1D
# weights (as QuadraticSpline below): 3 × n instead of n³ × 3 1D evaluations.

"""Generic stencil weights: evaluated node by node with `shapefunction`."""
struct NodewiseWeights{S <: AbstractShapeFunction, T}
    spline::S
    grid_coords::SVector{3, T}
    base::NTuple{3, Int}
end

"""
    stencil_weights(spline, grid_coords, base)

Weights of all support nodes of a particle at `grid_coords` (`base` from
`get_support_base`). Use with `stencil_weight(W, di, dj, dk)`.
"""
@inline stencil_weights(spline::AbstractShapeFunction, grid_coords::SVector{3}, base) =
    NodewiseWeights(spline, grid_coords, (Int(base[1]), Int(base[2]), Int(base[3])))

"""
    stencil_weight(W, di, dj, dk)

Weight of the support node `base + (di, dj, dk)`, identical to
`shapefunction(spline, grid_coords - node)`.
"""
@inline function stencil_weight(W::NodewiseWeights, di, dj, dk)
    node = SVector(W.base[1] + di, W.base[2] + dj, W.base[3] + dk)
    return shapefunction(W.spline, W.grid_coords - node)
end

# QuadraticSpline: table of the 1D weights, W[k, d] = weight along axis k of the
# node base[k] + d - 1. Same natural coordinates (exact subtractions) and the same
# product as shapefunction; 9 instead of 81 evaluations of quadspline_1d.
@inline function stencil_weights(::QuadraticSpline, grid_coords::SVector{3, T}, base) where {T}
    g1, g2, g3 = grid_coords
    b1, b2, b3 = base
    return SMatrix{3, 3, T, 9}(
        quadspline_1d(g1 - b1),       quadspline_1d(g2 - b2),       quadspline_1d(g3 - b3),
        quadspline_1d(g1 - (b1 + 1)), quadspline_1d(g2 - (b2 + 1)), quadspline_1d(g3 - (b3 + 1)),
        quadspline_1d(g1 - (b1 + 2)), quadspline_1d(g2 - (b2 + 2)), quadspline_1d(g3 - (b3 + 2)))
end

@inline stencil_weight(W::SMatrix{3, 3}, di, dj, dk) = W[1, di + 1] * W[2, dj + 1] * W[3, dk + 1]

# APIC Logic
@inline function B_update(::QuadraticSpline, N, r_rel, v_I)
    return N * (v_I * r_rel')
end

@inline function M_inv(::QuadraticSpline, inv_dx)
    return 4 * inv_dx^2
end


