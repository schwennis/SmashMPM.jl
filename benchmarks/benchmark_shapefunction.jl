# File: benchmarks/benchmark_shapefunction.jl

"""
    add_shapefunction_benchmarks!(suite, T=Float64)

Register the shape function benchmarks in `suite["shapefunctions_T"]`.
These helpers are called 27 times per particle and transfer, so they must be
allocation-free.
"""
function add_shapefunction_benchmarks!(suite::BenchmarkGroup, ::Type{T}=Float64) where {T}
    grp = suite["shapefunctions_$(T)"] = BenchmarkGroup()

    spline     = QuadraticSpline()
    pos_p      = SVector{3, T}(0.37, -0.14, 0.82)
    origin     = SVector{3, T}(-1.0, -1.0, -1.0)
    inv_dx     = T(20.0)
    grid_pos   = SmashMPM.grid_position(pos_p, inv_dx, origin)
    nat_coords = SVector{3, T}(0.2, -0.6, 0.4)
    N_val      = shapefunction(spline, nat_coords)
    r_rel      = -nat_coords * (one(T) / inv_dx)
    v_I        = SVector{3, T}(12.5, -3.2, 0.8)

    grp["shapefunction_3d"] = @benchmarkable shapefunction($spline, $nat_coords)
    grp["grid_position"]    = @benchmarkable SmashMPM.grid_position($pos_p, $inv_dx, $origin)
    grp["get_support_base"] = @benchmarkable SmashMPM.get_support_base($spline, $grid_pos)
    grp["APIC_B_update"]    = @benchmarkable SmashMPM.B_update($spline, $N_val, $r_rel, $v_I)
    expect_no_allocs!(grp)

    return grp
end
