# ---------------------------------------------------------------------------- #
#                     AFLIP(0) reproduces APIC bit for bit                     #
# ---------------------------------------------------------------------------- #
# With α = 0 every FLIP term is an exact zero (0 * finite = 0, x + 0 = x), so the
# runtime FLIP path must give exactly the same numbers as the compile-time APIC
# path. Compared with `==` after every step: particles (pos, vel, C, F, mat_state,
# wave_speed), grid read state (mass, momentum, force) and dt.
# Cases cover both grids, the boundary condition in the G2P and the force field.

function _aflip_test_bodies()
    T   = Float64
    mat = NeoHookean(ρ = 1000.0, E = 1e6, ν = 0.3)
    return (
        Body(Sphere(0.3, SVector(-0.4, 0.0, 0.0)), SVector( 1.0, 0.0, 0.0), SVector(0.0, 0.0, 2.0), mat),
        Body(Sphere(0.3, SVector( 0.4, 0.3, 0.0)), SVector(-1.0, 0.0, 0.0), zero(SVector{3, T}),   mat),
    )
end

function _aflip_test_model(scheme; kw...)
    setup = SimulationSetup(; dx = 0.1, t_max = 1.0, cfl_number = 0.4, dt_max = 1e-3,
                              seed = 0, transfer_scheme = scheme, kw...)
    return build_mpm_model(_aflip_test_bodies(), setup)
end

_aflip_grid_fields(s) = (s.mass,
                         s.momentum.x, s.momentum.y, s.momentum.z,
                         s.force.x,    s.force.y,    s.force.z)

_aflip_n_active(::DenseGrid)  = 0
_aflip_n_active(g::SparseGrid) = g.n_active_read

"""First mismatching quantity between two models, or `nothing` if bitwise equal."""
function _aflip_first_mismatch(a, b)
    for (n, (pa, pb)) in enumerate(zip(a.particle_sets, b.particle_sets))
        P, Q = pa.particles, pb.particles
        P.pos       == Q.pos       || return "set $n: pos"
        P.vel       == Q.vel       || return "set $n: vel"
        P.C         == Q.C         || return "set $n: C"
        P.F         == Q.F         || return "set $n: F"
        P.mat_state == Q.mat_state || return "set $n: mat_state"
        P.wave_speed == Q.wave_speed || return "set $n: wave_speed"
    end
    _aflip_n_active(a.grid) == _aflip_n_active(b.grid) || return "grid: n_active_read"
    names = (:mass, :momentum_x, :momentum_y, :momentum_z, :force_x, :force_y, :force_z)
    for (name, fa, fb) in zip(names, _aflip_grid_fields(a.grid.state_read),
                                     _aflip_grid_fields(b.grid.state_read))
        fa == fb || return "grid: $name"
    end
    return nothing
end

const _AFLIP_CASES = (
    (label = "DenseGrid, open",
     kw    = (; grid_type = DenseGrid)),
    (label = "DenseGrid, NoSlip + gravity",
     kw    = (; grid_type = DenseGrid, boundary_condition = NoSlipBoundary(),
                external_force = ConstantGravity(SVector(0.0, -9.81, 0.0)))),
    (label = "SparseGrid, OpenDomain",
     kw    = (; grid_type = SparseGrid, domain = OpenDomain())),
    (label = "SparseGrid, FittedBox + NoSlip + gravity",
     kw    = (; grid_type = SparseGrid, domain = FittedBox(), boundary_condition = NoSlipBoundary(),
                external_force = ConstantGravity(SVector(0.0, -9.81, 0.0)))),
)

@testset "AFLIP(0) reproduces APIC" begin
    n_steps = 30
    @testset "$(c.label)" for c in _AFLIP_CASES
        apic  = _aflip_test_model(APIC();     c.kw...)
        aflip = _aflip_test_model(AFLIP(0.0); c.kw...)

        @test _aflip_first_mismatch(apic, aflip) === nothing      # identical start

        mismatch, dt_equal = nothing, true
        for s in 1:n_steps
            dt_a = SmashMPM.step!(apic)
            dt_b = SmashMPM.step!(aflip)
            dt_equal &= (dt_a == dt_b)
            m = _aflip_first_mismatch(apic, aflip)
            if m !== nothing || !dt_equal
                mismatch = "step $s: " * something(m, "dt")
                break
            end
        end

        @test dt_equal
        @test mismatch === nothing
        mismatch === nothing || @info "AFLIP(0) vs APIC differs" c.label mismatch

        # sanity: the simulation really moved, and the spheres touched (stress was active)
        @test apic.t > 0
        @test any(F -> F != one(F), apic.particle_sets[1].particles.F)
    end
end