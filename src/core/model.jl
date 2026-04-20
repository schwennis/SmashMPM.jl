mutable struct Model{
    T, 
    MGT<:AbstractVector{<:MaterialGroup}, 
    GT<:AbstractGrid, MT<:AbstractMaterial,
    BC<:AbstractBoundaryCondition, 
    S<:AbstractSolver, 
    EF<:AbstractExternalForce
}

    material_groups::MGT
    grid::GT
    boundary_condition::BC
    solver::S
    external_force::EF

    t::T
    t_max::T
end