# ---------------------------------------------------------------------------- #
#          HDF5 export: particles/... and grid/... layout, sparse tiles         #
# ---------------------------------------------------------------------------- #
using HDF5

function _hdf5_test_models()
    T   = Float64
    mat = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
    bodies = (
        Body(Sphere(0.3, SVector(-0.4, 0.0, 0.0)), SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 2.0), mat),
        Body(Sphere(0.3, SVector( 0.4, 0.1, 0.0)), SVector(-1.0, 0.0, 0.0), zero(SVector{3, T}), mat),
    )
    setup(; kw...) = SimulationSetup(dx=0.1, t_max=1.0; kw...)

    models = (
        dense       = build_mpm_model(bodies, setup()),
        sparse_open = build_mpm_model(bodies, setup(grid_type=SparseGrid)),
        sparse_box  = build_mpm_model(bodies, setup(grid_type=SparseGrid, domain=FittedBox())),
    )
    for m in models, _ in 1:3
        SmashMPM.step!(m)
    end
    return models
end

function _read_grid_export(path)
    h5open(path, "r") do f
        g = f["grid"]
        return (
            node_index = read(g["node_index"]),
            position   = read(g["position"]),
            mass       = read(g["mass"]),
            momentum   = read(g["momentum"]),
            force      = read(g["force"]),
            origin     = read(g["origin"]),
            dx         = read(g["dx"]),
            attrs      = Dict(k => read(attributes(g)[k]) for k in keys(attributes(g))),
        )
    end
end

# Dataset names of a group (as a Dict, so `haskey` reads naturally)
f_keys(path, group) = h5open(f -> Dict(k => true for k in keys(f[group])), path, "r")

# Active part of a sparse state field (columns 1:n_active_read)
_active(grid, field) = view(field, :, 1:grid.n_active_read)

@testset "HDF5 export of the grid and particles" begin
    models = _hdf5_test_models()
    dir    = mktempdir()

    try
        ref_mass = sum(models.dense.grid.state_read.mass)
        ref_px   = sum(models.dense.grid.state_read.momentum.x)
        ref_py   = sum(models.dense.grid.state_read.momentum.y)

        # ------------------------------------------------------------------ #
        @testset "SparseGrid (OpenDomain): only the active tiles, no dense box" begin
            model = models.sparse_open
            g     = model.grid

            for level in (0, 3)
                exporter = HDF5Exporter(output_dir=dir, filename_prefix="open$(level)_",
                                        export_grid=true, compression_level=level)
                path = write_output(exporter, model, 7)
                out  = _read_grid_export(path)

                n_tiles = g.n_active_read
                M       = SmashMPM.TILE_VOLUME * n_tiles

                # exactly the nodes of the active tiles
                @test out.attrs["type"] == "SparseGrid"
                @test out.attrs["n_tiles"] == n_tiles
                @test out.attrs["tile_size"] == SmashMPM.TILE_SIZE
                @test out.attrs["n_nodes"] == M
                @test size(out.node_index) == (3, M)
                @test size(out.position) == (3, M)
                @test size(out.momentum) == (3, M)
                @test length(out.mass) == M
                @test size(out.force) == (3, M)
                @test !haskey(f_keys(path, "grid"), "wave_speed")     # wave speeds are particle data now

                # nodes are unique and group into exactly n_tiles tiles
                ijk = out.node_index
                @test length(Set(Tuple.(eachcol(ijk)))) == M
                tiles = Set(zip(SmashMPM.tile_coord.(ijk[1, :]),
                                SmashMPM.tile_coord.(ijk[2, :]),
                                SmashMPM.tile_coord.(ijk[3, :])))
                @test length(tiles) == n_tiles

                # reproducible order: tiles sorted by key
                first_nodes = ijk[:, 1:SmashMPM.TILE_VOLUME:M]
                keys_in_file = [SmashMPM.tile_key(SmashMPM.tile_coord(c[1]),
                                                  SmashMPM.tile_coord(c[2]),
                                                  SmashMPM.tile_coord(c[3])) for c in eachcol(first_nodes)]
                @test issorted(keys_in_file)

                # positions belong to the node indices
                err = maximum(norm(out.position[:, n] .- SmashMPM.node_position(g.origin, g.inv_dx,
                                   ijk[1, n], ijk[2, n], ijk[3, n])) for n in 1:M)
                @test err < 1e-12
                @test out.dx ≈ 1 / g.inv_dx
                @test out.origin ≈ collect(g.origin)

                # nodal data = active part of the read state
                @test sum(out.mass)          ≈ sum(_active(g, g.state_read.mass))
                @test sum(out.momentum[1, :]) ≈ sum(_active(g, g.state_read.momentum.x)) atol = 1e-10
                @test maximum(abs, out.force) ≈ maximum(maximum(abs, _active(g, f))
                                                        for f in (g.state_read.force.x,
                                                                  g.state_read.force.y,
                                                                  g.state_read.force.z))

                # same totals as the DenseGrid run
                @test sum(out.mass)           ≈ ref_mass rtol = 1e-12
                @test sum(out.momentum[1, :]) ≈ ref_px   atol = 1e-10
                @test sum(out.momentum[2, :]) ≈ ref_py   atol = 1e-10
            end
        end

        # ------------------------------------------------------------------ #
        @testset "SparseGrid (BoxDomain): nodes stay inside the box" begin
            model = models.sparse_box
            g     = model.grid
            path  = write_output(HDF5Exporter(output_dir=dir, filename_prefix="box_", export_grid=true), model, 7)
            out   = _read_grid_export(path)
            M     = out.attrs["n_nodes"]

            @test out.attrs["n_tiles"] == g.n_active_read
            @test M <= SmashMPM.TILE_VOLUME * g.n_active_read
            @test all(all(g.domain.lo[d] .<= out.node_index[d, :] .<= g.domain.hi[d]) for d in 1:3)
            @test sum(out.mass) ≈ ref_mass rtol = 1e-12
            @test sum(out.momentum[1, :]) ≈ ref_px atol = 1e-10
        end

        # ------------------------------------------------------------------ #
        @testset "DenseGrid: all nodes, same layout" begin
            model = models.dense
            g     = model.grid
            path  = write_output(HDF5Exporter(output_dir=dir, filename_prefix="dense_", export_grid=true), model, 7)
            out   = _read_grid_export(path)
            dims  = size(g.state_read.mass)

            @test out.attrs["type"] == "DenseGrid"
            @test !haskey(out.attrs, "n_tiles")
            @test out.attrs["n_nodes"] == prod(dims)
            @test reshape(out.mass, dims) ≈ g.state_read.mass
            @test reshape(out.momentum[1, :], dims) ≈ g.state_read.momentum.x
            @test out.node_index[:, 1] == [1, 1, 1]
            @test out.node_index[:, 2] == [2, 1, 1]         # i runs fastest
            @test out.node_index[:, end] == collect(dims)
            @test out.position[:, 1] ≈ collect(g.origin)
        end

        # ------------------------------------------------------------------ #
        @testset "one group / ParaView block per particle set" begin
            model = models.sparse_open
            sets  = model.particle_sets
            n_part = sum(ps -> length(ps.particles.mass), sets)

            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="both_", export_grid=true), model, 1)
            h5open(path, "r") do f
                @test Set(keys(f)) ⊇ Set(["particles", "grid"])
                @test Set(keys(f["particles"])) == Set(["ParticleSet$(i)" for i in eachindex(sets)])
                @test read(attributes(f["particles"])["n_particles"]) == n_part
                @test read(attributes(f["particles"])["n_sets"]) == length(sets)
                @test read(attributes(f)["step"]) == 1

                for (i, ps) in enumerate(sets)
                    g = f["particles/ParticleSet$(i)"]
                    N = length(ps.particles.mass)
                    @test read(attributes(g)["n_particles"]) == N
                    @test read(attributes(g)["set_index"]) == i
                    @test read(attributes(g)["material"]) == "NeoHookean"

                    @test read(g["position"]) ≈ vcat(ps.particles.pos.x', ps.particles.pos.y', ps.particles.pos.z')
                    @test size(read(g["velocity"])) == (3, N)
                    @test size(read(g["affine"])) == (9, N)
                    @test size(read(g["F"])) == (9, N)
                    @test size(read(g["stress"])) == (9, N)
                    @test read(g["mass"]) ≈ ps.particles.mass
                    @test read(g["id"]) == Int.(ps.particles.id)
                    @test all(==(i), read(g["set_index"]))
                    @test !haskey(g, "von_mises")                 # not a default field
                    @test !haskey(g, "mat_state")                 # NeoHookean has no material state

                    # velocities are the G2P of the read grid, per set
                    v, _ = SmashMPM.extract_velocities(model.grid, ps, model.shapefunction)
                    @test read(g["velocity"]) ≈ reduce(hcat, v)
                end
                ws = reduce(vcat, [read(f["particles/ParticleSet$(i)/wave_speed"]) for i in eachindex(sets)])
                @test maximum(ws) == SmashMPM.max_wavespeed(model)
                # no flat particle datasets anymore
                @test !haskey(f["particles"], "position") && !("position" in keys(f))
            end

            xmf = read(replace(path, ".h5" => ".xmf"), String)
            for i in eachindex(sets)
                @test occursin("<Grid Name=\"ParticleSet$(i)\"", xmf)
                @test occursin("$(basename(path)):/particles/ParticleSet$(i)/position", xmf)
                @test occursin("$(basename(path)):/particles/ParticleSet$(i)/stress", xmf)
            end
            @test occursin("<Grid Name=\"MPM_Grid\"", xmf)
            @test occursin("$(basename(path)):/grid/position", xmf)
            @test occursin("$(basename(path)):/grid/momentum", xmf)
            @test occursin("$(basename(path)):/grid/force", xmf)
            @test occursin("Name=\"Stress\" AttributeType=\"Tensor\"", xmf)
            @test occursin("Name=\"Velocity\" AttributeType=\"Vector\"", xmf)
            @test !occursin("MPM_Particles", xmf)
            @test !occursin("3DCoRectMesh", xmf)

            # grid only
            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="gridonly_",
                                             export_particles=false, export_grid=true), model, 2)
            h5open(path, "r") do f
                @test !haskey(f, "particles")
                @test haskey(f, "grid")
            end
            @test !occursin("ParticleSet", read(replace(path, ".h5" => ".xmf"), String))

            # particles only (default)
            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="partonly_"), model, 3)
            h5open(path, "r") do f
                @test haskey(f, "particles")
                @test !haskey(f, "grid")
            end
            @test !occursin("MPM_Grid", read(replace(path, ".h5" => ".xmf"), String))
        end

        # ------------------------------------------------------------------ #
        @testset "stress = material model of the stored state" begin
            model = models.dense
            path  = write_output(HDF5Exporter(output_dir=dir, filename_prefix="stress_",
                                              particle_fields=(:stress, :von_mises)), model, 4)
            h5open(path, "r") do f
                for (i, ps) in enumerate(model.particle_sets)
                    mat = ps.material
                    σ_ref = map(ps.particles.F) do F            # NeoHookean, by hand
                        J = det(F)
                        (mat.μ * (F * F' - I) + mat.λ * log(J) * I) / J
                    end
                    σ = read(f["particles/ParticleSet$(i)/stress"])
                    @test σ ≈ SmashMPM._matrices(σ_ref)
                    @test any(!iszero, σ)
                    # row-major layout, symmetric anyway: σ[2,1] == σ[1,2]
                    @test σ[2, :] ≈ σ[4, :]
                    vm = read(f["particles/ParticleSet$(i)/von_mises"])
                    @test vm ≈ SmashMPM.von_mises.(σ_ref)
                    @test all(>=(0), vm)
                end
            end
        end

        # ------------------------------------------------------------------ #
        @testset "material state fields are written per set" begin
            T     = Float64
            basalt = Basalt(T, eos=:murnaghan, elasticity=:hypoelastic, damage=:no_damage)
            neo    = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)
            bodies = (
                Body(Sphere(0.3, SVector(-0.4, 0.0, 0.0)), SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 5.0), basalt),
                Body(Sphere(0.3, SVector( 0.4, 0.0, 0.0)), zero(SVector{3, T}), zero(SVector{3, T}), neo),
            )
            model = build_mpm_model(bodies, SimulationSetup(dx=0.1, t_max=1.0, grid_type=SparseGrid))
            for _ in 1:3
                SmashMPM.step!(model)
            end

            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="solid_", export_grid=true), model, 0)
            h5open(path, "r") do f
                g1 = f["particles/ParticleSet1"]
                @test read(attributes(g1)["material"]) == "SolidMaterial"
                fields = keys(g1["mat_state"])
                @test "c" in fields
                @test "e" in fields
                @test "eos_state" in fields
                @test "elastic_state" in fields
                @test "p" in keys(g1["mat_state/eos_state"])

                # σ = s - p I for the hypoelastic solid without damage
                s = read(g1["mat_state/elastic_state/s"])
                p = read(g1["mat_state/eos_state/p"])
                σ = read(g1["stress"])
                @test σ ≈ s .- [1, 0, 0, 0, 1, 0, 0, 0, 1] .* p'  atol = 1e-8 * max(1, maximum(abs, σ))

                # the NeoHookean set has no material state: no NaN padding anymore
                @test !haskey(f["particles/ParticleSet2"], "mat_state")
            end
            xmf = read(replace(path, ".h5" => ".xmf"), String)
            @test occursin("$(basename(path)):/particles/ParticleSet1/mat_state/c", xmf)
            @test occursin("Name=\"mat_state.eos_state.p\"", xmf)
            @test !occursin("ParticleSet2/mat_state", xmf)

            # selection of single material state fields
            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="solidsel_",
                                             particle_fields=(:mass, "mat_state/eos_state")), model, 0)
            h5open(path, "r") do f
                g = f["particles/ParticleSet1"]
                @test Set(keys(g)) == Set(["position", "mass", "mat_state"])
                @test Set(keys(g["mat_state"])) == Set(["eos_state"])
            end
        end

        # ------------------------------------------------------------------ #
        @testset "field selection" begin
            model = models.sparse_open
            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="sel_",
                                             particle_fields=(:mass,), export_grid=true,
                                             grid_fields=(:mass, :velocity)), model, 5)
            h5open(path, "r") do f
                @test Set(keys(f["particles/ParticleSet1"])) == Set(["position", "mass"])
                @test Set(keys(f["grid"])) == Set(["position", "mass", "velocity", "origin", "dx"])
                m = read(f["grid/mass"]); v = read(f["grid/velocity"])
                g = model.grid
                @test sum(m .* v[1, :]) ≈ sum(_active(g, g.state_read.momentum.x)) atol = 1e-10
            end
            xmf = read(replace(path, ".h5" => ".xmf"), String)
            @test !occursin("/particles/ParticleSet1/velocity", xmf)
            @test occursin("/grid/velocity", xmf)

            # :all / :default / single symbol
            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="all_", particle_fields=:all), model, 6)
            h5open(path, "r") do f
                @test haskey(f["particles/ParticleSet1"], "von_mises")
                @test haskey(f["particles/ParticleSet1"], "stress")
            end
            ex = HDF5Exporter(particle_fields=(:default, :von_mises))
            @test ex.particle_fields.names == Set([SmashMPM.DEFAULT_PARTICLE_FIELDS[1:end-1]..., :von_mises])
            @test ex.particle_fields.mat_state == ["mat_state"]
            @test HDF5Exporter(particle_fields=:stress).particle_fields.names == Set([:stress])
            @test HDF5Exporter(particle_fields=["position", "F"]).particle_fields.names == Set([:F])

            # typos fail early, when the exporter is created
            @test_throws ArgumentError HDF5Exporter(particle_fields=(:stres,))
            @test_throws ArgumentError HDF5Exporter(grid_fields=(:stress,))
            @test_throws ArgumentError HDF5Exporter(grid_fields=("mat_state/e",))
        end
    finally
        rm(dir; recursive=true, force=true)
    end
end