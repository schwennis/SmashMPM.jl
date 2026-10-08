using BenchmarkTools
using StaticArrays

include("../src/SmashMPM.jl")
using .SmashMPM

F = @SMatrix [1.05 0.02 0.00;
			  0.00 0.98 0.01;
			  0.00 0.00 1.02]
C = @SMatrix [0.10 0.02 0.00;
			  0.00 -0.05 0.01;
			  0.00 0.00 0.02]
V0 = 1.0
m = 7874.0
dt = 1.0e-6
dx = 0.01

neohookean = NeoHookean(E=200.0e9, ν=0.3, ρ=7874.0)
neohookean_state = initial_material_state(neohookean)

iron_tillotson = Iron(Float64, eos=:tillotson, elasticity=:hyperelastic)
iron_tillotson_state = initial_material_state(iron_tillotson)

iron_murnaghan = Iron(Float64, eos=:murnaghan, elasticity=:hyperelastic)
iron_murnaghan_state = initial_material_state(iron_murnaghan)

println("Benchmarking NeoHookean...")
material_model(neohookean, neohookean_state, F, C, V0, m, dt, dx)
display(@benchmark material_model(
	$neohookean, $neohookean_state, $F, $C, $V0, $m, $dt, dx
))

println("Benchmarking Iron with Tillotson EoS...")
material_model(iron_tillotson, iron_tillotson_state, F, C, V0, m, dt, dx)
display(@benchmark material_model(
	$iron_tillotson, $iron_tillotson_state, $F, $C, $V0, $m, $dt, dx
))

println("Benchmarking Iron with Murnaghan EoS...")
material_model(iron_murnaghan, iron_murnaghan_state, F, C, V0, m, dt, dx)
display(@benchmark material_model(
	$iron_murnaghan, $iron_murnaghan_state, $F, $C, $V0, $m, $dt, dx
))
