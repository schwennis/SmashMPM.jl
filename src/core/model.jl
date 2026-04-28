mutable struct Model{
    T, 
    MGT<:Tuple{<:AbstractMaterialGroup}, 
    GT<:AbstractGrid,
    BC<:AbstractBoundaryCondition, 
    EF<:AbstractExternalForce,
    SF<:AbstractShapeFunction
}

    material_groups::MGT
    grid::GT
    boundary_condition::BC
    external_force::EF
    shapefunction::SF

    t::T
    t_max::T
end