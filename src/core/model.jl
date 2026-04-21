mutable struct Model{
    T, 
    MGT<:AbstractVector{<:MaterialGroup}, 
    GT<:AbstractGrid, MT<:AbstractMaterial,
    BC<:AbstractBoundaryCondition, 
    EF<:AbstractExternalForce
}

    material_groups::MGT
    grid::GT
    boundary_condition::BC
    external_force::EF

    t::T
    t_max::T
end