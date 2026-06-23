using BenchmarkTools
using InteractiveUtils
using Profile
using PProf
using StaticArrays

include("../src/SmashMPM.jl")
using .SmashMPM

spline = QuadraticSpline()

shapefunction(spline, SVector{3, Float64}(0.2, -0.1, 0.4)) # Testaufruf zum Vorcompilieren

println("Benchmarking shapefunction for QuadraticSpline...")
display(@benchmark shapefunction($spline, $(SVector{3, Float64}(rand(3)...))))