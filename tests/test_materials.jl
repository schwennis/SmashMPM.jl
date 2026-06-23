@testset "Fallback Behavior" begin
    # Wir definieren ein Dummy-Material, für das es keine Methode gibt
    struct DummyMaterial <: AbstractMaterial end
    dummy_mat = DummyMaterial()
    
    # Die Platzhalter-Variablen spielen hier keine Rolle, da sofort ein Fehler fliegen soll
    @test_throws ErrorException material_model(dummy_mat, nothing, nothing, nothing, nothing, nothing, nothing)
end

@testset "NeoHookean" begin
    include("materials/test_neohookean.jl")
end

