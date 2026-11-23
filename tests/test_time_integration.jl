# ---------------------------------------------------------------------------- #
#                  Time integration: kick-drift-kick (leapfrog)                #
# ---------------------------------------------------------------------------- #
# step! uses
#   x^{n+1}   = x^n + dt_n v^{n+1/2}                       (g2p2g, drift)
#   v^{n+3/2} = v^{n+1/2} + (dt_n + dt_{n+1})/2 * a^{n+1}   (grid_kick!, kick)
# with the half kick v^{1/2} = v^0 + dt_0/2 a^0 in build_mpm_model.
#
# For a constant acceleration this is EXACT, also with a variable time step:
# the sum of the kick steps up to step n is t_n + dt_n/2, hence
#   x_cm(t_n) = x_cm(0) + v0 t_n + g t_n^2 / 2
# up to rounding. A missing or wrong initial half kick gives an error that grows
# linearly in t (≈ g dt_0 t / 2), which is several percent of the displacement here.

function _ti_center_of_mass(model)
    M = 0.0
    X = zero(SVector{3, Float64})
    for ps in model.particle_sets
        p = ps.particles
        for i in eachindex(p.mass)
            M += p.mass[i]
            X += p.mass[i] * SVector{3, Float64}(p.pos.x[i], p.pos.y[i], p.pos.z[i])
        end
    end
    return X / M
end

function _ti_build(; grid_type, t_max=1.0, external_force=NoExternalForce(), v0)
    T     = Float64
    shape = RectangularPrism{T}(width=0.4, height=0.4, depth=0.4)
    mat   = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
    body  = Body(shape, v0, zero(SVector{3, T}), mat)
    setup = SimulationSetup(dx=0.1, t_max=t_max, grid_type=grid_type, external_force=external_force)
    return build_mpm_model((body,), setup)
end

@testset "Free fall: KDK is exact for constant acceleration" begin
    g  = SVector(0.0, -9.81, 0.0)
    v0 = SVector(0.5, 0.0, 0.0)
    n_steps = 20

    for grid_type in (DenseGrid, SparseGrid)
        @testset "$grid_type" begin
            model = _ti_build(; grid_type, external_force=ConstantGravity(g), v0)
            x0 = _ti_center_of_mass(model)

            dts = Float64[]
            for _ in 1:n_steps
                dt_expected = model.dt_previous         # step n uses the dt chosen at the end of step n-1
                dt = step!(model)
                @test dt == dt_expected
                push!(dts, dt)
            end

            t = model.t
            @test t ≈ sum(dts) rtol = 1e-14

            # the time step really varies (|v| enters the CFL wave speed)
            @test maximum(dts) > minimum(dts)

            disp_expected = v0 * t + g * t^2 / 2
            disp          = _ti_center_of_mass(model) - x0
            @test norm(disp - disp_expected) <= 1e-8 * norm(disp_expected)

            # sanity check that the test can detect a missing half kick
            dt0 = dts[1]
            @test norm(g) * dt0 * t / 2 > 1e-3 * norm(disp_expected)
        end
    end

    @testset "SparseGrid and DenseGrid follow the same trajectory" begin
        dense  = _ti_build(; grid_type=DenseGrid,  external_force=ConstantGravity(g), v0)
        sparse = _ti_build(; grid_type=SparseGrid, external_force=ConstantGravity(g), v0)
        for _ in 1:n_steps
            @test step!(dense) ≈ step!(sparse)
        end
        @test norm(_ti_center_of_mass(dense) - _ti_center_of_mass(sparse)) < 1e-12
    end
end

@testset "Uniform motion without forces" begin
    v0 = SVector(0.3, -0.2, 0.1)
    for grid_type in (DenseGrid, SparseGrid)
        model = _ti_build(; grid_type, v0)
        x0 = _ti_center_of_mass(model)
        for _ in 1:10
            step!(model)
        end
        @test norm(_ti_center_of_mass(model) - x0 - v0 * model.t) <= 1e-10 * norm(v0) * model.t
    end
end

@testset "Last step is clipped to t_max" begin
    v0 = SVector(0.5, 0.0, 0.0)
    for grid_type in (DenseGrid, SparseGrid)
        model = _ti_build(; grid_type, t_max=1e-2, v0)
        n = 0
        while model.t < model.t_max && n < 10_000
            step!(model)
            n += 1
        end
        @test n > 1
        @test model.t ≈ model.t_max rtol = 1e-12
        @test model.dt_previous ≈ 0.0 atol = 1e-12    # no time left for a further step
    end
end
