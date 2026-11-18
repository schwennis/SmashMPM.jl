#!/usr/bin/env -S julia --startup-file=no
# ---------------------------------------------------------------------------- #
#                         Conservation check (CLI tool)                        #
# ---------------------------------------------------------------------------- #
# Mass, linear momentum, angular momentum and energy of all particles over the
# HDF5 output of a run. Run `tools/conservation_check.jl --help` for the options.
#
# Can also be included to use the functions directly on a model:
#   include("tools/conservation_check.jl"); q = conserved_quantities(model)
#
# Angular momentum includes the affine (APIC/AFLIP) part, only the sum is conserved:
#   L_p = x_p × m_p v_p + m_p d (C₃₂-C₂₃, C₁₃-C₃₁, C₂₁-C₁₂),  d = Δx²/4 (quadratic), Δx²/3 (cubic)
# Likewise the kinetic energy: K_p = ½ m_p |v_p|² + ½ m_p d ‖C_p‖²  (≥ the grid kinetic energy)
# ---------------------------------------------------------------------------- #
# Started as a script from SmashMPM.jl/tools/ without --project: use the SmashMPM.jl
# project. Anywhere else (copied into a simulation folder) or with --project: keep the
# active environment.
import Pkg
if abspath(PROGRAM_FILE) == (@__FILE__) && Base.JLOptions().project == C_NULL
    let toml = joinpath(@__DIR__, "..", "Project.toml")
        if isfile(toml) && occursin(r"^name\s*=\s*\"SmashMPM\""m, read(toml, String))
            Pkg.activate(dirname(toml); io = devnull)
        end
    end
end

using SmashMPM
using HDF5
using LinearAlgebra
using StaticArrays
using Printf

const USAGE = """
usage: conservation_check.jl [DIR] [options]

Conservation of mass, momentum, angular momentum and energy over the HDF5 particle
output (export_particles = true) of a SmashMPM run.

arguments:
  DIR                  output directory (default: output)

options:
  -p, --prefix PREFIX  file name prefix (default: the only prefix found in DIR)
  --dx DX              node spacing; only needed for files without the `dx` attribute
  --spline NAME        quadratic | cubic; only needed for files without the `shapefunction` attribute
  --every N            only every N-th file (first and last are always checked)
  --csv FILE           also write the table as CSV
  --tol TOL            exit with code 1 if |ΔM|/M, |ΔP|/Σm|v| or |ΔL|/Σm|x||v| exceed TOL
  -h, --help           show this help

columns (relative to the first file):
  ΔM/M, |ΔP|/Σm|v|, |ΔL|/Σm|x||v|   should stay at round-off (no external forces / walls)
  E_kin/E0  ½ m |v|²,   E_aff/E0  ½ m d ‖C‖² (affine part, belongs to the kinetic energy)
  E_int/E0  m e,        E/E0 = sum. E/E0 < 1: numerical dissipation of the transfer
                                    (E_kin from half-step velocities: small wiggles are normal;
                                    no gravitational energy, no strain energy without mat_state/e)
"""

# ---------------------------------------------------------------------------- #
#                                   Totals                                     #
# ---------------------------------------------------------------------------- #
_spin(C) = SVector(C[3, 2] - C[2, 3], C[1, 3] - C[3, 1], C[2, 1] - C[1, 2])

"""
    conserved_quantities(pos, vel, C, mass, e, d)

Totals over particles. `d` is the spline factor of the affine angular momentum
(Δx²/4 quadratic, Δx²/3 cubic). `e` (specific internal energy) may be `nothing`.
"""
function conserved_quantities(pos, vel, C, mass, e, d)
    M = 0.0; P = zero(SVector{3, Float64})
    L_orb = zero(SVector{3, Float64}); L_spin = zero(SVector{3, Float64})
    E_kin = 0.0; E_aff = 0.0; E_int = 0.0
    P_scale = 0.0; L_scale = 0.0          # scales for relative errors (P or L may be ~0)
    n_nonfinite = 0
    for i in eachindex(mass)
        m = Float64(mass[i]); x = SVector{3, Float64}(pos[i]); v = SVector{3, Float64}(vel[i])
        Ci = SMatrix{3, 3, Float64, 9}(C[i])
        if !(isfinite(m) && all(isfinite, x) && all(isfinite, v) && all(isfinite, Ci))
            n_nonfinite += 1
        end
        M      += m
        P      += m * v
        L_orb  += m * cross(x, v)
        L_spin += m * d * _spin(Ci)
        E_kin  += 0.5 * m * dot(v, v)
        E_aff  += 0.5 * m * d * sum(abs2, Ci)       # kinetic energy of the affine velocity C r
        e === nothing || (E_int += m * Float64(e[i]))
        P_scale += m * norm(v)
        L_scale += m * norm(x) * norm(v)
    end
    return (; M, P, L = L_orb + L_spin, L_orb, L_spin, E_kin, E_aff, E_int, E = E_kin + E_aff + E_int,
              P_scale, L_scale, n_nonfinite, n = length(mass))
end

function spline_factor(name::AbstractString, dx)
    n = lowercase(name)
    occursin("quadratic", n) && return dx^2 / 4
    occursin("cubic", n)     && return dx^2 / 3
    error("unknown spline '$name' (use quadratic or cubic)")
end
spline_factor(s::SmashMPM.AbstractShapeFunction, dx) = spline_factor(string(nameof(typeof(s))), dx)

"""
    conserved_quantities(model)

Same, directly from an `MPMModel` (any backend, copies to the CPU).
"""
function conserved_quantities(model)
    cpu = SmashMPM.model_to_cpu(model)
    d   = spline_factor(cpu.shapefunction, 1 / cpu.grid.inv_dx)
    gather(f) = reduce(vcat, [vec(collect(f(ps.particles))) for ps in cpu.particle_sets])
    e = gather(p -> map(s -> hasproperty(s, :e) ? s.e : 0.0, p.mat_state))
    return conserved_quantities(gather(p -> p.pos), gather(p -> p.vel), gather(p -> p.C),
                                gather(p -> p.mass), e, d)
end

# ---------------------------------------------------------------------------- #
#                                HDF5 output                                   #
# ---------------------------------------------------------------------------- #
_vec3(A) = [SVector{3, Float64}(A[1, i], A[2, i], A[3, i]) for i in axes(A, 2)]
# affine is stored row-major (9, N): [C11 C12 C13 C21 ...]
_mat3(A) = [SMatrix{3, 3, Float64, 9}(A[1, i], A[4, i], A[7, i], A[2, i], A[5, i], A[8, i],
                                      A[3, i], A[6, i], A[9, i]) for i in axes(A, 2)]

function read_quantities(path; dx = nothing, spline = nothing)
    h5open(path, "r") do f
        a = attributes(f)
        t = read(a["time"])
        haskey(f, "particles") || error("$path: no particle data (export_particles = false?)")
        Δx = dx !== nothing ? dx :
             haskey(a, "dx")       ? read(a["dx"]) :
             haskey(f, "grid/dx")  ? read(f["grid/dx"]) :
             error("$path: no dx attribute (old file?), pass --dx")
        sp = spline !== nothing ? spline :
             haskey(a, "shapefunction") ? read(a["shapefunction"]) : "quadratic"
        p = f["particles"]
        e = haskey(p, "mat_state/e") ? read(p["mat_state/e"]) : nothing
        q = conserved_quantities(_vec3(read(p["position"])), _vec3(read(p["velocity"])),
                                 _mat3(read(p["affine"])), read(p["mass"]), e, spline_factor(sp, Δx))
        return t, q
    end
end

function find_files(dir, prefix)
    isdir(dir) || error("directory not found: $dir")
    h5 = sort(filter(f -> endswith(f, ".h5"), readdir(dir)))
    if prefix === nothing
        prefixes = unique(replace.(h5, r"\d+\.h5$" => ""))
        isempty(prefixes) && error("no .h5 files in $dir")
        length(prefixes) == 1 || error("several prefixes in $dir ($(join(prefixes, ", "))), pass --prefix")
        prefix = only(prefixes)
    end
    files = filter(f -> startswith(f, prefix) && occursin(r"^\d+\.h5$", f[length(prefix)+1:end]), h5)
    isempty(files) && error("no files $(prefix)<step>.h5 in $dir")
    return prefix, [joinpath(dir, f) for f in files]
end

# ---------------------------------------------------------------------------- #
#                                     CLI                                      #
# ---------------------------------------------------------------------------- #
function parse_args(args)
    opts = Dict{Symbol, Any}(:dir => "output", :prefix => nothing, :dx => nothing, :spline => nothing,
                             :every => 1, :csv => nothing, :tol => nothing)
    i = 1
    value(name) = (i < length(args) || error("missing value for $name"); i += 1; args[i])
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help");    print(USAGE); exit(0)
        elseif a in ("-p", "--prefix"); opts[:prefix] = value(a)
        elseif a == "--dx";          opts[:dx]     = parse(Float64, value(a))
        elseif a == "--spline";      opts[:spline] = value(a)
        elseif a == "--every";       opts[:every]  = max(1, parse(Int, value(a)))
        elseif a == "--csv";         opts[:csv]    = value(a)
        elseif a == "--tol";         opts[:tol]    = parse(Float64, value(a))
        elseif startswith(a, "-");   error("unknown option $a (see --help)")
        else                         opts[:dir]    = a
        end
        i += 1
    end
    return opts
end

function main(args)
    opts = parse_args(args)
    prefix, files = find_files(opts[:dir], opts[:prefix])
    sel = [k for k in eachindex(files) if (k - 1) % opts[:every] == 0 || k == length(files)]
    println("$(length(sel)) of $(length(files)) files $(prefix)*.h5 in $(opts[:dir])\n")

    header = ("t", "step", "ΔM/M", "|ΔP|/Σm|v|", "|ΔL|/Σm|x||v|", "E_kin/E0", "E_aff/E0", "E_int/E0", "E/E0")
    @printf("%-11s %7s %11s %11s %13s %10s %10s %10s %10s\n", header...)
    rows = []
    q0 = nothing; worst = 0.0; nonfinite_at = nothing
    for k in sel
        t, q = read_quantities(files[k]; dx = opts[:dx], spline = opts[:spline])
        q0 === nothing && (q0 = q)
        step = parse(Int, match(r"(\d+)\.h5$", files[k])[1])
        r = (t, step, (q.M - q0.M) / q0.M, norm(q.P - q0.P) / q0.P_scale, norm(q.L - q0.L) / q0.L_scale,
             q.E_kin / q0.E, q.E_aff / q0.E, q.E_int / q0.E, q.E / q0.E)
        push!(rows, r)
        @printf("%-11s %7d % .3e % .3e   % .3e %10.6f %10.6f %10.6f %10.6f%s\n",
                SmashMPM.format_sim_time(t), r[2:end]...,
                q.n_nonfinite > 0 ? "   ← $(q.n_nonfinite) non-finite particles" : "")
        if q.n_nonfinite > 0 && nonfinite_at === nothing
            nonfinite_at = t
        end
        all(isfinite, r[3:5]) ? (worst = max(worst, abs(r[3]), r[4], r[5])) : (worst = Inf)
    end

    if opts[:csv] !== nothing
        open(opts[:csv], "w") do io
            println(io, "time,step,dM_rel,dP_rel,dL_rel,Ekin_rel,Eaff_rel,Eint_rel,E_rel")
            for r in rows
                println(io, join(r, ","))
            end
        end
        println("\nwritten: $(opts[:csv])")
    end

    println()
    nonfinite_at === nothing || println("first non-finite particles at t = $(SmashMPM.format_sim_time(nonfinite_at))")
    @printf("max relative drift of M, P, L: %.3e\n", worst)
    if opts[:tol] !== nothing && !(worst <= opts[:tol])
        @printf("FAILED: drift exceeds tolerance %.1e\n", opts[:tol])
        exit(1)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    try
        main(ARGS)
    catch err
        err isa Base.IOError && err.code == Base.UV_EPIPE && exit(0)   # output piped into head etc.
        err isa ErrorException || rethrow()
        println(stderr, "error: ", err.msg)
        exit(2)
    end
end
