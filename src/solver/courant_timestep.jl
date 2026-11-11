# ---------------------------------------------------------------------------- #
#                               Courant Timestep                               #
# ---------------------------------------------------------------------------- #
# The wave speeds c_p + |v_p| are stored per particle by the last G2P2G
# (initially by build_mpm_model), so this is a plain reduction, no grid field.
max_wavespeed(model::MPMModel) = max_wavespeed(model.particle_sets)

function courant_timestep(model::MPMModel{T}) where {T} 
    dt_courant = model.cfl_number / (model.grid.inv_dx * max_wavespeed(model))
    
    return min(dt_courant, model.dt_max)
end