using BenchmarkTools
using InteractiveUtils
using Profile
using PProf
using StaticArrays

include("../src/SmashMPM.jl")
using .SmashMPM

mat = NeoHookean(λ=10.0, μ=5.0, ρ=1.0)
state = get_initial_material_state(mat)

# Dummy values for parameters currently ignored by your NeoHookean model
C_dummy = one(SMatrix{3,3,Float64,9})
V0_dummy = 1.0
m_dummy = 1.0
dt_dummy = 0.1
F_id = one(SMatrix{3,3,Float64,9})

# Precompile material_model
material_model(mat, state, F_id, C_dummy, V0_dummy, m_dummy, dt_dummy)

# View code stability
# @code_warntype material_model(mat, state, F_id, C_dummy, V0_dummy, m_dummy, dt_dummy)



println("Benchmarking material_model for NeoHookean material...")
display(@benchmark material_model($mat, $state, $F_id, $C_dummy, $V0_dummy, $m_dummy, $dt_dummy))

Profile.clear_malloc_data()
for i in 1:1000
    material_model(mat, state, F_id, nothing, 1.0, 1.0, 0.1)
end