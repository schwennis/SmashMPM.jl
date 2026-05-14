function step!(model::Model, cfl_factor::T=0.4) where {T}
    grid = model.grid

    grid_reset!(grid)

    dt = courant_timestep(model, cfl_factor)

    for material_group in model.material_groups
        g2p2g!(grid, material_group, model.shapefunction, dt)
    end

    apply_external_forces!(grid, model.external_force, dt)

    apply_boundary_condition!(grid, model.boundary_condition)
end