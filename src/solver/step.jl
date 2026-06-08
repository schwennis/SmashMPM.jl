function step!(model::Model, cfl_factor::T=0.4) where {T}
    grid = model.grid

    grid_reset!(grid)

    dt = courant_timestep(model, cfl_factor)

    for material_group in model.material_groups
        g2p2g!(grid, material_group, model.shapefunction, dt)
    end

    apply_external_forces!(model.external_force, grid, dt)

    apply_boundary_condition!(model.boundary_condition, grid)
end