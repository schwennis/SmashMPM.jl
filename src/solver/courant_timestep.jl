function courant_timestep(model::Model, cfl_factor::T=0.5) where {T} 
    # Extract necessary information from the model
    grid = model.grid.state_old
   
    dt_courant = cfl_factor / (grid.inv_dx * max_wavespeed(grid))
    
    return dt_courant
end