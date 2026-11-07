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
            wave_speed = read(g["wave_speed"]),
            origin     = read(g["origin"]),
            dx         = read(g["dx"]),
            attrs      = Dict(k => read(attributes(g)[k]) for k in keys(attributes(g))),
        )
    end
end

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
                @test length(out.wave_speed) == M

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
                @test maximum(out.wave_speed) ≈ maximum(_active(g, g.state_read.wave_speed))

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
        @testset "Particles and grid share the same layout; options" begin
            model   = models.sparse_open
            n_part  = sum(ps -> length(ps.particles.mass), model.particle_sets)
            m_part  = sum(ps -> sum(ps.particles.mass), model.particle_sets)

            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="both_", export_grid=true), model, 1)
            h5open(path, "r") do f
                @test Set(keys(f)) ⊇ Set(["particles", "grid"])
                @test size(read(f["particles/position"])) == (3, n_part)
                @test size(read(f["particles/velocity"])) == (3, n_part)
                @test size(read(f["particles/F"])) == (9, n_part)
                @test sum(read(f["particles/mass"])) ≈ m_part
                @test read(attributes(f["particles"])["n_particles"]) == n_part
                @test read(attributes(f)["step"]) == 1
                # no flat datasets in the file root anymore
                @test !("position" in keys(f)) && !("mass" in keys(f))
            end

            xmf = read(replace(path, ".h5" => ".xmf"), String)
            @test occursin("$(basename(path)):/particles/position", xmf)
            @test occursin("$(basename(path)):/grid/position", xmf)
            @test occursin("$(basename(path)):/grid/momentum", xmf)
            @test !occursin("3DCoRectMesh", xmf)

            # grid only
            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="gridonly_",
                                             export_particles=false, export_grid=true), model, 2)
            h5open(path, "r") do f
                @test !haskey(f, "particles")
                @test haskey(f, "grid")
            end

            # particles only (default)
            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="partonly_"), model, 3)
            h5open(path, "r") do f
                @test haskey(f, "particles")
                @test !haskey(f, "grid")
            end
            @test !occursin("MPM_Grid", read(replace(path, ".h5" => ".xmf"), String))
        end

        # ------------------------------------------------------------------ #
        @testset "material state fields are written under particles/mat_state" begin
            T     = Float64
            basalt = Basalt(T, eos=:murnaghan, elasticity=:hypoelastic)
            body   = Body(Sphere(0.3, zero(SVector{3, T})), zero(SVector{3, T}), zero(SVector{3, T}), basalt)
            model  = build_mpm_model((body,), SimulationSetup(dx=0.1, t_max=1.0, grid_type=SparseGrid))

            path = write_output(HDF5Exporter(output_dir=dir, filename_prefix="solid_", export_grid=true), model, 0)
            h5open(path, "r") do f
                fields = keys(f["particles/mat_state"])
                @test "c" in fields
                @test "e" in fields
                @test "eos_state" in fields
                @test "elastic_state" in fields
                @test "p" in keys(f["particles/mat_state/eos_state"])
            end
            @test occursin("$(basename(path)):/particles/mat_state/c", read(replace(path, ".h5" => ".xmf"), String))
        end
    finally
        rm(dir; recursive=true, force=true)
    end
end