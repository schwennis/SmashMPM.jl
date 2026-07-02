abstract type AbstractShapeFunction end

# ---------------------------------------------------------------------------- #
#                                    Helpers                                   #
# ---------------------------------------------------------------------------- #
@inline function get_grid_position(pos_p::SVector{3, T}, inv_dx::T, origin::SVector{3, T}, padding::Int) where {T}
    return (pos_p - origin) * inv_dx .+ (padding + one(T))  # +1 for 1-based indexing
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

    if abs_d < 0.5
        # Fall 1: |x| < 0.5
        # N(x) = 0.75 - x^2
        w = T(0.75) - abs_d^2
    elseif abs_d < 1.5
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

# APIC Logic
@inline function B_update(::QuadraticSpline, N, r_rel, v_I)
    return N * (v_I * r_rel')
end

@inline function M_inv(::QuadraticSpline, inv_dx)
    return 4 * inv_dx^2
end


