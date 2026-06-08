struct ConstantGravity{T}<:AbstractExternalForce
    g::SVector{3, T}
end

function apply_external_forces!(force::ConstantGravity{T}, grid::DenseGrid{T, S}, dt::T) where {T, S}
    state = grid.state_new

    state.p .+= state.m .* force.g .* dt
end