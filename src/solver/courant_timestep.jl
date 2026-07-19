# ---------------------------------------------------------------------------- #
#                               Courant Timestep                               #
# ---------------------------------------------------------------------------- #
function courant_timestep(model::MPMModel, cfl_factor::T=0.5) where {T} 
    # Extract necessary information from the model
    grid = model.grid
   
    dt_courant = cfl_factor / (grid.inv_dx * max_wavespeed(grid))
    
    return min(dt_courant, model.dt_max)
end


