# File: benchmarks/run_benchmarks.jl
#
# Usage:
#   julia --threads=auto --color=yes benchmarks/run_benchmarks.jl [options]
#
# Options:
#   --quick            smaller problems and a shorter time budget
#   --cpu-only         do not load CUDA
#   --float64          run in Float64 (default: Float32)
#   --filter=<str>     only run benchmarks whose path contains <str>
#   --save-baseline    save the results as new baseline (benchmarks/baseline.json)
#   --strict           exit with code 1 on time regressions or unexpected allocations
#   --speedup          additionally run the CPU vs. GPU scaling benchmark
#   --speedup-only     only run the CPU vs. GPU scaling benchmark
#
# Allocation column: green = allocation-free as required, red = allocations that
# should not have happened, uncolored = allocations are not checked.
# (Colors need a terminal or `--color=yes`.)
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using BenchmarkTools
using KernelAbstractions
using StaticArrays
using Printf
using SmashMPM

include("bench_utils.jl")
include("benchmark_materials.jl")
include("benchmark_shapefunction.jl")
include("benchmark_grid.jl")
include("benchmark_boundary_conditions.jl")
include("benchmark_external_forces.jl")
include("benchmark_solver.jl")
include("benchmark_speedup.jl")

# ---------------------------------------------------------------------------- #
#                            Command Line Arguments                            #
# ---------------------------------------------------------------------------- #
function parse_cli_args(args)
    filter_str = ""
    for a in args
        startswith(a, "--filter=") && (filter_str = split(a, "=", limit=2)[2])
    end

    return (;
        save_baseline = "--save-baseline" in args || "--save" in args,
        quick         = "--quick" in args,
        cpu_only      = "--cpu-only" in args,
        strict        = "--strict" in args,
        speedup       = "--speedup" in args,
        speedup_only  = "--speedup-only" in args,
        T             = "--float64" in args ? Float64 : Float32,
        filter_str,
        baseline_file = joinpath(@__DIR__, "baseline.json"),
    )
end


# ---------------------------------------------------------------------------- #
#                               Benchmark Suite                                #
# ---------------------------------------------------------------------------- #
"""
    build_suite(backends; T=Float32, quick=false) -> BenchmarkGroup

Scalar benchmarks (backend independent) go into `suite["scalar"]`, everything
else into one group per backend (`suite["CPU"]`, `suite["CUDA"]`).
"""
function build_suite(backends; T::Type{<:AbstractFloat}=Float32, quick::Bool=false)
    suite = BenchmarkGroup()

    # 1. Scalar / register level (backend independent)
    scalar = suite["scalar"] = BenchmarkGroup()
    add_material_benchmarks!(scalar, T)
    add_shapefunction_benchmarks!(scalar, T)

    # 2. Backend specific (CPU and optionally CUDA)
    n_side      = quick ? 32 : 64
    n_sparse    = quick ? 10_000 : 25_000
    solver_cnts = quick ? [10_000] : [10_000, 50_000]

    for (b_name, backend) in backends
        b_grp = suite[b_name] = BenchmarkGroup()
        add_grid_benchmarks!(b_grp, backend; T=T, n_side=n_side, n_particles_sparse=n_sparse)
        add_boundary_condition_benchmarks!(b_grp, backend; T=T, n_side=n_side, n_particles_sparse=n_sparse)
        add_external_force_benchmarks!(b_grp, backend; T=T, n_side=n_side, n_particles_sparse=n_sparse)
        add_solver_benchmarks!(b_grp, backend; T=T, particle_counts=solver_cnts)
    end

    return suite
end

"""Keep only the benchmarks whose path contains `pattern`. Tags are preserved."""
function filter_suite(grp::BenchmarkGroup, pattern::AbstractString, prefix::String="")
    out = similar(grp)
    for (key, val) in grp
        path = isempty(prefix) ? string(key) : "$(prefix)/$(key)"
        if val isa BenchmarkGroup
            sub = filter_suite(val, pattern, path)
            isempty(sub.data) || (out[key] = sub)
        elseif occursin(lowercase(pattern), lowercase(path))
            out[key] = val
        end
    end
    return out
end

"""
    configure_suite!(suite; seconds)

Time budget for all benchmarks. Scalar benchmarks are tuned (many evaluations per
sample, they take only nanoseconds); all others use one evaluation per sample,
since their `setup` has to run before every evaluation.
"""
function configure_suite!(suite::BenchmarkGroup; seconds::Real)
    for (path, b) in BenchmarkTools.leaves(suite)
        b.params.seconds = seconds
        first(path) == "scalar" || (b.params.evals = 1)
    end
    haskey(suite, "scalar") && tune!(suite["scalar"])
    return suite
end


# ---------------------------------------------------------------------------- #
#                           Flattening & Formatting                            #
# ---------------------------------------------------------------------------- #
"""A benchmark result together with its path and allocation expectation."""
struct FlatTrial
    path::String
    trial::BenchmarkTools.Trial
    no_allocs::Bool     # true if the benchmark must be allocation-free
end

function flatten_trials(grp::BenchmarkGroup, prefix::String="")
    entries = FlatTrial[]
    for key in sort!(collect(keys(grp)); by=string)
        val  = grp[key]
        path = isempty(prefix) ? string(key) : "$(prefix) / $(key)"
        if val isa BenchmarkGroup
            append!(entries, flatten_trials(val, path))
        elseif val isa BenchmarkTools.Trial
            push!(entries, FlatTrial(path, val, expects_no_allocs(grp, key)))
        end
    end
    return entries
end

function format_time(ns::Real)
    ns < 1e3 && return @sprintf("%7.2f ns", ns)
    ns < 1e6 && return @sprintf("%7.2f μs", ns / 1e3)
    ns < 1e9 && return @sprintf("%7.2f ms", ns / 1e6)
    return @sprintf("%7.2f s ", ns / 1e9)
end

function format_memory(bytes::Real)
    bytes < 1024   && return @sprintf("%7d B  ", bytes)
    bytes < 1024^2 && return @sprintf("%7.2f KiB", bytes / 1024)
    return @sprintf("%7.2f MiB", bytes / 1024^2)
end

format_speedup(s::Real) = @sprintf("%7.2f×", s)

"""Cut long paths from the left, the end of the path is the most specific part."""
_fit(s::AbstractString, n::Int) = length(s) <= n ? s : "…" * last(s, n - 1)


# ---------------------------------------------------------------------------- #
#                               Allocation Check                               #
# ---------------------------------------------------------------------------- #
"""
    alloc_status(entry) -> :ok | :unexpected | :unchecked

`:ok` and `:unexpected` only for benchmarks marked with `expect_no_allocs!`.
"""
function alloc_status(e::FlatTrial)
    e.no_allocs || return :unchecked
    return allocs(e.trial) == 0 ? :ok : :unexpected
end

function _print_alloc(io::IO, str::AbstractString, status::Symbol)
    if status === :ok
        printstyled(io, str; color=:green)
    elseif status === :unexpected
        printstyled(io, str; color=:red, bold=true)
    else
        print(io, str)
    end
end

function _print_alloc_legend(io::IO)
    print(io, "Allocs: ")
    printstyled(io, "green"; color=:green)
    print(io, " = allocation-free as required, ")
    printstyled(io, "red"; color=:red, bold=true)
    println(io, " = unexpected allocations, uncolored = not checked")
end

"""
    check_allocations(results; io=stdout) -> Bool

Lists every benchmark that must be allocation-free but allocated.
"""
function check_allocations(results::BenchmarkGroup; io::IO=stdout)
    flat      = flatten_trials(results)
    n_checked = count(e -> e.no_allocs, flat)
    bad       = filter(e -> alloc_status(e) === :unexpected, flat)

    println(io)
    if isempty(bad)
        printstyled(io, "All $(n_checked) allocation-free benchmarks are allocation-free.\n"; color=:green)
    else
        printstyled(io, "$(length(bad)) of $(n_checked) allocation-free benchmarks allocated:\n";
                    color=:red, bold=true)
        for e in bad
            printstyled(io, "  $(e.path): $(allocs(e.trial)) allocs, $(strip(format_memory(memory(e.trial))))\n";
                        color=:red)
        end
    end
    return isempty(bad)
end


# ---------------------------------------------------------------------------- #
#                                Result Tables                                 #
# ---------------------------------------------------------------------------- #
function print_results_table(results::BenchmarkGroup; io::IO=stdout)
    width = 104
    println(io, "\n", "="^width)
    @printf(io, "%-62s | %-11s | %10s | %11s\n", "Benchmark", "Median", "Allocs", "Memory")
    println(io, "-"^width)
    for e in flatten_trials(results)
        est = median(e.trial)
        @printf(io, "%-62s | %-11s | ", _fit(e.path, 62), format_time(est.time))
        _print_alloc(io, @sprintf("%10d | %11s", est.allocs, format_memory(est.memory)), alloc_status(e))
        println(io)
    end
    println(io, "="^width)
    _print_alloc_legend(io)
    return nothing
end

"""
    print_backend_speedups(results; reference="CPU", io=stdout)

Per-kernel speedup of every accelerator group in `results` relative to `reference`.
"""
function print_backend_speedups(results::BenchmarkGroup; reference::String="CPU", io::IO=stdout)
    haskey(results, reference) || return nothing
    ref = Dict(e.path => e.trial for e in flatten_trials(results[reference]))

    for name in sort!(collect(keys(results)); by=string)
        (name == reference || name == "scalar") && continue
        acc = Dict(e.path => e.trial for e in flatten_trials(results[name]))

        width = 100
        println(io, "\n", "="^width)
        println(io, " $(reference) vs. $(name) SPEEDUP PER KERNEL")
        println(io, "-"^width)
        @printf(io, "%-56s | %-11s | %-11s | %-9s\n", "Kernel / Operation", "$(reference)", "$(name)", "Speedup")
        println(io, "-"^width)
        for key in sort!(collect(keys(ref)))
            haskey(acc, key) || continue
            t_ref = median(ref[key]).time
            t_acc = median(acc[key]).time
            @printf(io, "%-56s | %-11s | %-11s | %-9s\n",
                    _fit(key, 56), format_time(t_ref), format_time(t_acc), format_speedup(t_ref / t_acc))
        end
        println(io, "="^width)
    end
    return nothing
end


# ---------------------------------------------------------------------------- #
#                             Baseline Comparison                              #
# ---------------------------------------------------------------------------- #
"""
    compare_with_baseline(results, baseline_file; time_tol=0.10, io=stdout) -> Bool

Judges the median times against a saved baseline. Returns false on regressions.
"""
function compare_with_baseline(results::BenchmarkGroup, baseline_file::String; time_tol=0.10, io::IO=stdout)
    if !isfile(baseline_file)
        @info "No baseline found at '$(baseline_file)'. Run with `--save-baseline` to create one."
        return true
    end

    baseline  = BenchmarkTools.load(baseline_file)[1]
    curr_flat = Dict(e.path => e.trial for e in flatten_trials(results))
    base_flat = Dict(e.path => e.trial for e in flatten_trials(baseline))

    width = 105
    println(io, "\n", "="^width)
    println(io, " REGRESSION CHECK AGAINST BASELINE (tolerance: ±$(round(Int, 100 * time_tol)) %)")
    println(io, "-"^width)
    @printf(io, "%-52s | %-12s | %-12s | %-9s | %-10s\n", "Benchmark", "Current", "Baseline", "Ratio", "Status")
    println(io, "-"^width)

    n_regressions = 0
    for key in sort!(collect(keys(curr_flat)))
        haskey(base_flat, key) || continue
        med_curr = median(curr_flat[key])
        med_base = median(base_flat[key])
        verdict  = judge(med_curr, med_base; time_tolerance=time_tol, memory_tolerance=0.05)

        status = if verdict.time == :regression
            n_regressions += 1
            "REGRESSION"
        elseif verdict.time == :improvement
            "FASTER"
        else
            "OK"
        end

        @printf(io, "%-52s | %-12s | %-12s | %8.2f× | %-10s\n",
                _fit(key, 52), format_time(med_curr.time), format_time(med_base.time),
                med_curr.time / med_base.time, status)
    end
    println(io, "="^width)

    if n_regressions > 0
        @warn "$(n_regressions) performance regression(s) compared to the baseline!"
    else
        @info "All benchmarks within the tolerance or faster than the baseline."
    end
    return n_regressions == 0
end


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
"""Build, run and evaluate the benchmark suite. Returns false on regressions or unexpected allocations."""
function run_benchmark_suite(cfg, backends)
    println("Building benchmark suite (precision: $(cfg.T), quick mode: $(cfg.quick))...")
    suite = build_suite(backends; T=cfg.T, quick=cfg.quick)

    if !isempty(cfg.filter_str)
        @info "Filter active: '$(cfg.filter_str)'"
        suite = filter_suite(suite, cfg.filter_str)
    end
    configure_suite!(suite; seconds=cfg.quick ? 0.4 : 1.5)

    println("Running benchmarks...")
    results = run(suite; verbose=true)

    print_results_table(results)
    print_backend_speedups(results)
    allocs_ok = check_allocations(results)

    time_ok = true
    if cfg.save_baseline
        BenchmarkTools.save(cfg.baseline_file, results)
        @info "New baseline saved to: $(cfg.baseline_file)"
    else
        time_ok = compare_with_baseline(results, cfg.baseline_file)
    end
    return allocs_ok && time_ok
end

function main(args=ARGS)
    cfg      = parse_cli_args(args)
    backends = detect_backends(; cpu_only=cfg.cpu_only)
    # CUDA may have just been loaded: run everything else in the latest world,
    # otherwise the CUDA methods (allocate, kernels, ...) are not visible.
    return Base.invokelatest(_main, cfg, backends)
end

function _main(cfg, backends)
    passed = true

    if !cfg.speedup_only
        passed &= run_benchmark_suite(cfg, backends)
        foreach(((_, backend),) -> reclaim_device_memory!(backend), backends)
    end

    if cfg.speedup || cfg.speedup_only
        run_speedup_benchmarks(
            backends;
            T=cfg.T,
            particle_counts=cfg.quick ? [10_000, 100_000] : [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
            seconds=cfg.quick ? 0.5 : 2.0,
        )
    end

    cfg.strict && !passed && exit(1)
    return passed
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
