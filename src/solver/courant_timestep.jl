# ---------------------------------------------------------------------------- #
#                               Courant Timestep                               #
# ---------------------------------------------------------------------------- #
function courant_timestep(model::MPMModel{T}) where {T} 
    # Extract necessary information from the model
    grid = model.grid
   
    dt_courant = model.cfl_number / (grid.inv_dx * max_wavespeed(grid))
    
    return min(dt_courant, model.dt_max)
end


