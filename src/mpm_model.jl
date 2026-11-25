@kwdef mutable struct MPMModel{
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
    boundary_condition::BC = NullBoundaryCondition()
    external_force::EF = NoExternalForce()
    shapefunction::SF = QuadraticSpline()

    backend::B = CPU()

    t::T = 0.0
    t_max::T
    dt_max::T = 0.001
    dt_previous::T = 0.0
    cfl_number::T = 0.4
end



function model_to_cpu(model::MPMModel)
    if model.backend isa CPU
        return model
    end

    cpu_particle_sets = map(model.particle_sets) do p_set
        SoAParticleSet(
            adapt(Array, p_set.particles), 
            adapt(Array, p_set.particles_buffer),
            p_set.material
        )
    end

    cpu_grid = grid_to_cpu(model.grid)

    cpu_model = MPMModel(
        particle_sets = cpu_particle_sets,
        grid = cpu_grid,
        boundary_condition = model.boundary_condition,
        external_force = model.external_force,
        shapefunction = model.shapefunction,
        backend = CPU(),
        t = model.t,
        t_max = model.t_max,
        dt_previous = model.dt_previous,
        dt_max = model.dt_max,
        cfl_number = model.cfl_number
    )
    return cpu_model
end


# ---------------------------------------------------------------------------- #
#                              Grid -> CPU copies                              #
# ---------------------------------------------------------------------------- #
function grid_to_cpu(grid::DenseGrid)
    return DenseGrid(
        adapt(Array, grid.state_read),
        adapt(Array, grid.state_write),
        grid.padding,
        grid.origin,
        grid.inv_dx
    )
end

# Stays sparse (a dense copy of an open domain can be arbitrarily large).
# Use to_dense(grid) explicitly where a DenseGrid is required.
function grid_to_cpu(grid::SparseGrid)
    return SparseGrid(
        adapt(Array, grid.state_read),
        adapt(Array, grid.state_write),
        adapt(Array, grid.map_read),
        adapt(Array, grid.map_write),
        grid.n_active_read,
        grid.n_active_write,
        grid.domain,
        grid.padding,
        grid.origin,
        grid.inv_dx
    )
end