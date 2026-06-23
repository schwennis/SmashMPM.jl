mutable struct MPMModel{
    T, 
    MGT<:Tuple, 
    GT<:AbstractGrid,
    BC<:AbstractBoundaryCondition, 
    EF<:AbstractExternalForce,
    SF<:AbstractShapeFunction,
    B
}

    particle_sets::MGT
    grid::GT
    boundary_condition::BC
    external_force::EF
    shapefunction::SF

    backend::B

    t::T
    t_max::T
    CFL_number::T
end