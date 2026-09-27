using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using CUDA
using KernelAbstractions
using StaticArrays
using Printf
using SmashMPM

CUDA.allowscalar(false)

function create_model(n_target::Int, backend::Backend; T=Float32)
    R = T(0.5)
    ppc_1d = 2
    sphere_vol = T(4/3 * π * R^3)
    spacing = T((sphere_vol / n_target)^(1/3))
    dx = spacing * ppc_1d

    mat = NeoHookean(ρ=T(1000.0), E=T(1e5), ν=T(0.3))
    shape = SmashMPM.Sphere(center=zero(SVector{3, T}), radius=R)
    v0 = SVector{3, T}(0.0, -10.0, 0.0)
    body = Body(shape, v0, zero(v0), mat)

    setup = SimulationSetup(
        dx=dx,
        t_max=T(1.0),
        padding=3,
        ppc_1d=ppc_1d,
        CFL_number=T(0.4),
        dt_max=T(1e-4),
        backend=backend
    )

    model = build_mpm_model((body,), setup)
    backend isa CUDABackend && CUDA.synchronize()
    return model
end

function benchmark_solver(model, backend::Backend; n_steps=50, n_warmup=10)
    T = eltype(model.grid.origin)
    dt = T(1e-4)
    sync() = backend isa CUDABackend ? CUDA.synchronize() : nothing

    # Warmup
    for _ in 1:n_warmup
        SmashMPM.g2p2g!(model, dt)
        SmashMPM.grid_reset!(model.grid)
    end
    sync()

    # Measurement
    t0 = time_ns()
    for _ in 1:n_steps
        SmashMPM.g2p2g!(model, dt)
        SmashMPM.grid_reset!(model.grid)
    end
    sync()

    elapsed_sec = (time_ns() - t0) / 1e9
    return (elapsed_sec / n_steps) * 1000.0
end

function run_cpu_gpu_comparison(targets; n_steps=50)
    println("-"^78)
    @printf("%-10s | %-14s | %-12s | %-12s | %-10s\n",
            "Particles", "Grid Dims", "CPU (ms)", "GPU (ms)", "Speedup")
    println("-"^78)

    for target in targets
        # Benchmark CPU
        model_cpu = create_model(target, CPU())
        np = length(model_cpu.particle_sets[1].particles.mass)
        dims = size(model_cpu.grid.state_old.mass)
        dims_str = "$(dims[1])×$(dims[2])×$(dims[3])"

        ms_cpu = benchmark_solver(model_cpu, CPU(); n_steps=n_steps)
        model_cpu = nothing
        GC.gc()

        # Benchmark GPU
        model_gpu = create_model(target, CUDABackend())
        ms_gpu = benchmark_solver(model_gpu, CUDABackend(); n_steps=n_steps)
        model_gpu = nothing
        GC.gc()
        CUDA.reclaim()

        speedup = ms_cpu / ms_gpu
        @printf("%-10d | %-14s | %-12.3f | %-12.3f | %-10.2f×\n",
                np, dims_str, ms_cpu, ms_gpu, speedup)
    end
    println("-"^78)
end

function main()
    @assert CUDA.functional() "CUDA device not available"
    dev = CUDA.device()
    println("CUDA Device : $(CUDA.name(dev))")
    println("CPU Threads : $(Threads.nthreads())")

    # Scaled problem sizes from cache-bound to GPU-saturated
    targets = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
    run_cpu_gpu_comparison(targets; n_steps=40)
end

main()