@testset "Fallback Behavior" begin
    struct DummyMaterial <: AbstractMaterial end
    @test_throws ErrorException material_model(DummyMaterial(), nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

# @testset "Legacy / Simple Materials" begin
#     include("materials/test_neohookean.jl")
# end

@testset "Equations of State (EoS)" begin
    include("materials/eos/test_murnaghan.jl")
    include("materials/eos/test_tillotson.jl")
end

@testset "Elasticity Models" begin
    include("materials/elasticity/test_elasticity.jl")
end

@testset "Viscosity Models" begin
    include("materials/viscosity/test_bulk_viscosity.jl")
end

@testset "Modular SolidMaterial Pipeline" begin
    include("materials/test_solidmaterial.jl")
end