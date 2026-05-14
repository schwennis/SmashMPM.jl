struct ConstantGravity{T}<:AbstractExternalForce
    g::SVector{3, T}
end

function apply_external_forces!(grid::DenseGrid{T, S}, force::ConstantGravity{T}, dt::T) where {T, S}
    state = grid.state_new

    state.p .+= state.m .* force.g .* dt
end