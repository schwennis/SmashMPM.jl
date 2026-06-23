include("../src/SmashMPM.jl")
using .SmashMPM

using Test
using JET
using LinearAlgebra
using StaticArrays
using StructArrays
using KernelAbstractions

@testset "SmashMPM" begin
    
    @testset "Materials" begin
        include("test_materials.jl")
    end

    @testset "Shape Functions" begin
        include("test_shapefunctions.jl")
    end

    @testset "Particles" begin
        include("test_particles.jl")
    end

    @testset "Grid" begin
        include("test_grid.jl")
    end

    @testset "Boundary Conditions" begin
        include("test_boundary_conditions.jl")
    end

    @testset "External Forces" begin
        include("test_ext_forces.jl")
    end

    @testset "MPM Model Builder Pipeline" begin
        include("test_sim_builder.jl")
    end
end