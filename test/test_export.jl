# ---------------------------------------------------------------------------- #
#                       Output: HDF5 / XDMF and JLD2 snapshots                  #
# ---------------------------------------------------------------------------- #
# The written data must be the model data: positions, masses, G2P velocities,
# the stress of the material model, the nodal fields of the read grid (with the
# same totals as the grid itself). A JLD2 snapshot must continue bit-identically.
using HDF5
using JLD2

function export_models()
    solid  = Basalt(Float64; eos = :murnaghan, elasticity = :hypoelastic, damage = :no_damage)
    neo    = NeoHookean(ρ = 1000.0, E = 1e6, ν = 0.3)
    bodies = (Body(Sphere(0.3, V3(-0.4, 0, 0)), V3(1, 0, 0), V3(0, 0, 5), solid),
              Body(Sphere(0.3, V3(0.4, 0.1, 0)), V3(-1, 0, 0), zero(V3), neo))
    setup(; kw...) = SimulationSetup(dx = 0.1, t_max = 1.0; kw...)
    models = (dense = build_mpm_model(bodies, setup()),
              open  = build_mpm_model(bodies, setup(grid_type = SparseGrid)),
              box   = build_mpm_model(bodies, setup(grid_type = SparseGrid, domain = FittedBox())))
    foreach(m -> run!(m, 3), models)
    return models
end

h5read_all(path, group) = h5open(f -> Dict(k => read(f[group][k]) for k in keys(f[group])), path)
h5attr(path, obj, name) = h5open(f -> read(attributes(obj == "/" ? f : f[obj])[name]), path)
xmf(path) = read(replace(path, ".h5" => ".xmf"), String)

@testset "HDF5 export" begin
    models = export_models()
    dir    = mktempdir()

    @testset "particle sets" begin
        model = models.open
        path  = write_output(HDF5Exporter(output_dir = dir, filename_prefix = "p_"), model, 7)
        @test h5attr(path, "/", "step") == 7 && h5attr(path, "/", "time") == model.t
        @test h5attr(path, "particles", "n_sets") == 2

        for (i, ps) in enumerate(model.particle_sets)
            p, d = ps.particles, h5read_all(path, "particles/ParticleSet$i")
            @test h5attr(path, "particles/ParticleSet$i", "material") == string(nameof(typeof(ps.material)))
            @test d["position"] == vcat(p.pos.x', p.pos.y', p.pos.z')
            @test d["mass"] == p.mass && d["id"] == Int.(p.id) && all(==(i), d["set_index"])
            @test d["F"] == S._matrices(p.F)
            v, _ = S.extract_velocities(model.grid, ps, model.shapefunction)
            @test d["velocity"] == reduce(hcat, v)                     # G2P of the read grid

            # stress = material model of the stored state (dt = 0, no viscosity)
            σ_ref = if ps.material isa NeoHookean
                map(F -> (ps.material.μ * (F * F' - I) + ps.material.λ * log(det(F)) * I) / det(F), p.F)
            else
                [st.elastic_state.s - st.eos_state.p * I for st in p.mat_state]
            end
            @test d["stress"] ≈ S._matrices(σ_ref) rtol = 1e-10            # (@fastmath det in NeoHookean)
            @test haskey(d, "mat_state") == (ps.material isa SolidMaterial)   # only real state fields
        end
        d1 = h5open(f -> Dict(k => read(f["particles/ParticleSet1/mat_state"][k]) for k in ("c", "e")), path)
        @test d1["e"] == [st.e for st in model.particle_sets[1].particles.mat_state]
        @test occursin("<Grid Name=\"ParticleSet1\"", xmf(path)) && occursin("<Grid Name=\"ParticleSet2\"", xmf(path))
    end

    @testset "grid nodes ($name)" for name in keys(models)
        model = models[name]
        g     = model.grid
        path  = write_output(HDF5Exporter(output_dir = dir, filename_prefix = "g$(name)_", export_grid = true,
                                          grid_fields = :all), model, 1)
        d     = h5read_all(path, "grid")
        n     = h5attr(path, "grid", "n_nodes")
        tot   = grid_totals(g)
        ijk   = d["node_index"]

        @test size(ijk) == (3, n) && allunique(eachcol(ijk))
        @test maximum(c -> norm(d["position"][:, c] - S.node_position(g.origin, g.inv_dx, ijk[:, c]...)), 1:n) < 1e-12
        @test sum(d["mass"]) ≈ tot.M rtol = 1e-13
        @test vec(sum(d["momentum"]; dims = 2)) ≈ tot.P rtol = 1e-12
        @test vec(sum(d["mass"]' .* d["velocity"]; dims = 2)) ≈ tot.P rtol = 1e-12
        if g isa DenseGrid
            @test n == length(g.state_read.mass) && reshape(d["mass"], size(g.state_read.mass)) == g.state_read.mass
        else
            @test n <= S.TILE_VOLUME * g.n_active_read                 # active tiles only, no dense box
        end
        if g isa SparseGrid && g.domain isa BoxDomain                  # boundary tiles stick out, their nodes not
            @test all(g.domain.lo .<= vec(minimum(ijk; dims = 2))) && all(vec(maximum(ijk; dims = 2)) .<= g.domain.hi)
        end
        @test occursin("<Grid Name=\"MPM_Grid\"", xmf(path))
    end

    @testset "field selection" begin
        model = models.dense
        path  = write_output(HDF5Exporter(output_dir = dir, filename_prefix = "sel_", particle_fields = (:mass,),
                                          export_grid = true, grid_fields = (:mass,)), model, 2)
        @test Set(keys(h5read_all(path, "particles/ParticleSet1"))) == Set(["position", "mass"])
        @test Set(keys(h5read_all(path, "grid"))) == Set(["position", "mass", "origin", "dx"])

        path = write_output(HDF5Exporter(output_dir = dir, filename_prefix = "ms_",
                                         particle_fields = ("mat_state/eos_state", :von_mises)), model, 3)
        d    = h5open(f -> Set(keys(f["particles/ParticleSet1"])), path)
        @test d == Set(["position", "von_mises", "mat_state"])
        @test h5open(f -> Set(keys(f["particles/ParticleSet1/mat_state"])), path) == Set(["eos_state"])
        @test !occursin("ParticleSet1/velocity", xmf(path))

        # typos fail when the exporter is created, not at the first output
        @test_throws ArgumentError HDF5Exporter(particle_fields = (:stres,))
        @test_throws ArgumentError HDF5Exporter(grid_fields = (:stress,))
    end
    rm(dir; recursive = true, force = true)
end

@testset "JLD2 snapshot continues the run" begin
    dir      = mktempdir()
    model    = colliding_blocks(; transfer = PolyPIC(), grid_type = SparseGrid)
    run!(model, 5)
    path     = write_output(S.JLD2Exporter(output_dir = dir, filename_prefix = "snap_"), model, 5)
    restored = JLD2.load(path, "model")
    run!(model, 5)
    run!(restored, 5)
    # identical up to the (thread order dependent) rounding of the atomic adds
    @test restored.t ≈ model.t rtol = 1e-14
    @test maximum(norm.(all_positions(restored) .- all_positions(model))) < 1e-12
    rm(dir; recursive = true, force = true)
end
