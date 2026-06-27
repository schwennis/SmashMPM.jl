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



function model_to_CPU(model::MPMModel)
    if model.backend isa CPU
        return model
    end

    cpu_particle_sets = map(model.particle_sets) do p_set
        SoAParticleSet(
            adapt(Array, p_set.particles), 
            p_set.material,
            CPU()
        )
    end

    old_state_cpu = adapt(Array, model.grid.state_old)
    new_state_cpu = adapt(Array, model.grid.state_new)

    cpu_grid = DenseGrid(
        old_state_cpu,
        new_state_cpu,
        model.grid.padding,
        model.grid.origin,
        model.grid.inv_dx
    )
    
    cpu_model = MPMModel(
        cpu_particle_sets,
        cpu_grid,
        model.boundary_condition,
        model.external_force,
        model.shapefunction,
        CPU(),
        model.t,
        model.t_max,
        model.CFL_number
    )
    return cpu_model
end