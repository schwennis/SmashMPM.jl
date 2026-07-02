abstract type AbstractExternalForce end

# ---------------------------------------------------------------------------- #
#                               No External FOrce                              #
# ---------------------------------------------------------------------------- #
struct NoExternalForce <: AbstractExternalForce end

function apply_external_forces!(force::NoExternalForce, grid, dt)
    return
end


# ---------------------------------------------------------------------------- #
#                               Constant Gravity                               #
# ---------------------------------------------------------------------------- #
struct ConstantGravity{T}<:AbstractExternalForce
    g::SVector{3, T}
end

function apply_external_forces!(force::ConstantGravity{T}, grid::DenseGrid{T, S}, dt::T) where {T, S}
    state = grid.state_new

    for i in 1:size(state.mass, 1), j in 1:size(state.mass, 2), k in 1:size(state.mass, 3)
        state.momentum[i, j, k] = state.momentum[i, j, k] .+ state.mass[i, j, k] .* force.g .* dt
    end
end


# ---------------------------------------------------------------------------- #
#                              Radial Force Field                              #
# ---------------------------------------------------------------------------- #
struct RadialInvSquareForceField{T} <: AbstractExternalForce
    # F(r) = F_0 * (r - center) / ||r - center||^2
    F_0::T
    center::SVector{3, T}
end

function apply_external_forces!(force::RadialInvSquareForceField{T}, grid::DenseGrid{T, S}, dt::T) where {T, S}
    state = grid.state_new

    for i in 1:size(state.mass, 1), j in 1:size(state.mass, 2), k in 1:size(state.mass, 3)
        pos = grid.origin .+ 1/ grid.inv_dx .* SVector(i-1, j-1, k-1)
        r_vec = pos .- force.center 
        r = norm(r_vec) + 1e-8
        force_vec = force.F_0 * (r_vec ./ r^2)
        state.momentum[i, j, k] = state.momentum[i, j, k] .+ state.mass[i, j, k] .* force_vec .* dt
    end
end


# ---------------------------------------------------------------------------- #
#                              Vector Field Force                              #
# ---------------------------------------------------------------------------- #
struct VectorFieldForce{A} <: AbstractExternalForce
    force_field::A  # Array{SVector}
end

function apply_external_forces!(force::VectorFieldForce{A}, grid::DenseGrid{T, S}, dt::T) where {A, T, S}
    state = grid.state_old
    force_field = force.force_field

    @assert size(force_field) == size(state.mass) "Force field dimensions must match grid dimensions."

    for i in 1:size(state.mass, 1), j in 1:size(state.mass, 2), k in 1:size(state.mass, 3)
        force_vec = force.force_field[i, j, k]
        state.momentum[i, j, k] = state.momentum[i, j, k] .+ state.mass[i, j, k] .* force_vec .* dt
    end 
end
