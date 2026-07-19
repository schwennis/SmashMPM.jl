using SmashMPM
using StaticArrays
using BenchmarkTools
using Profile
using LinearAlgebra
using Atomix: @atomic
using AMDGPU
using KernelAbstractions

# backend = ROCBackend()
backend = CPU()


center = SVector{3, Float64}(0.0, 0.0, 0.0)
R = 1.0
h = 1.0
euler_angles = SVector{3, Float64}(0.0, 0.0, 0.0)

shape1 = Cylinder(center, R, h, euler_angles)

linear_vel = SVector{3, Float64}(0.0, 0.0, 0.0)
rot_speed = 2π/5.0 
rotational_vec = SVector{3, Float64}(0.0, 0.0, rot_speed)
mat = NeoHookean(ρ=1000.0, E=1e6, ν=0.3)

body = Body(shape1, linear_vel, rotational_vec, mat)
bodies = (body,)

Setup = SimulationSetup(dx=0.1, t_end=1.0, backend=backend)

model = build_mpm_model(bodies, Setup)

N_particles = length(model.particle_sets[1].particles.pos)
dims = size(model.grid.state_old.mass)

println("Model built successfully with $N_particles particles and a grid of size $dims.")



g2p2g!(model, 1e-5)
display(@benchmark g2p2g!($model, 1e-5))

Profile.clear_malloc_data()

g2p2g!(model, 1e-5)