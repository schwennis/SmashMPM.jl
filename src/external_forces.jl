# ============================================================================ #
#                               EXTERNAL FORCES                                #
# ============================================================================ #

abstract type AbstractExternalForce end

# ---------------------------------------------------------------------------- #
#                               No External Force                              #
# ---------------------------------------------------------------------------- #
struct NoExternalForce <: AbstractExternalForce end

@inline function apply_external_forces!(::NoExternalForce, grid::DenseGrid, dt::Real)
    return nothing
end


# ---------------------------------------------------------------------------- #
#                               Constant Gravity                               #
# ---------------------------------------------------------------------------- #
struct ConstantGravity{T} <: AbstractExternalForce
    g::SVector{3, T}
end

function apply_external_forces!(force::ConstantGravity, grid::DenseGrid{T, S}, dt::Real) where {T, S}
    state = grid.state_new

    # Skalare vorab berechnen, um Multiplikationen im GPU-Loop zu sparen
    gx_dt = T(force.g[1] * dt)
    gy_dt = T(force.g[2] * dt)
    gz_dt = T(force.g[3] * dt)

    state.momentum.x .+= state.mass .* gx_dt
    state.momentum.y .+= state.mass .* gy_dt
    state.momentum.z .+= state.mass .* gz_dt
end


# ---------------------------------------------------------------------------- #
#                              Radial Force Field                              #
# ---------------------------------------------------------------------------- #
struct RadialInvSquareForceField{T} <: AbstractExternalForce
    F_0::T
    center::SVector{3, T}
end

@kernel function radial_inv_square_force_kernel!(momentum_x, momentum_y, momentum_z, mass, origin, inv_dx, center, F_0, dt)
    # Direkte 3D-Indexierung über GPU-Register (vermeidet div/mod)
    i, j, k = @index(Global, NTuple)

    m = mass[i, j, k]
    T = eltype(mass)

    # Nur auswerten, wenn am Knoten tatsächlich Masse anliegt
    if m > eps(T)
        dx = one(T) / inv_dx
        pos = SVector{3, T}(
            origin[1] + dx * (i - 1),
            origin[2] + dx * (j - 1),
            origin[3] + dx * (k - 1)
        )
        r_vec = pos .- center
        r = norm(r_vec) + eps(T)
        force_vec = T(F_0) * (r_vec ./ (r * r))

        momentum_x[i, j, k] += m * force_vec[1] * dt
        momentum_y[i, j, k] += m * force_vec[2] * dt
        momentum_z[i, j, k] += m * force_vec[3] * dt
    end
end

function apply_external_forces!(force::RadialInvSquareForceField, grid::DenseGrid{T, S}, dt::Real) where {T, S}
    state = grid.state_new

    backend = KernelAbstractions.get_backend(state.mass)
    kernel = radial_inv_square_force_kernel!(backend)

    kernel(
        state.momentum.x,
        state.momentum.y,
        state.momentum.z,
        state.mass,
        grid.origin,
        grid.inv_dx,
        force.center,
        force.F_0,
        T(dt);
        ndrange=size(state.mass) # 3D-Range
    )

    KernelAbstractions.synchronize(backend)
end


# ---------------------------------------------------------------------------- #
#                              Vector Field Force                              #
# ---------------------------------------------------------------------------- #
struct VectorFieldForce{A} <: AbstractExternalForce
    force_field::A
end

@kernel function vector_field_force_kernel!(momentum_x, momentum_y, momentum_z, mass, force_field, dt)
    i, j, k = @index(Global, NTuple)

    m = mass[i, j, k]
    if m > eps(eltype(mass))
        f_vec = force_field[i, j, k]
        momentum_x[i, j, k] += m * f_vec[1] * dt
        momentum_y[i, j, k] += m * f_vec[2] * dt
        momentum_z[i, j, k] += m * f_vec[3] * dt
    end
end

function apply_external_forces!(force::VectorFieldForce, grid::DenseGrid{T, S}, dt::Real) where {T, S}
    state = grid.state_new
    force_field = force.force_field

    @assert size(force_field) == size(state.mass) "Force field dimensions must match grid dimensions."

    backend = KernelAbstractions.get_backend(state.mass)
    
    # Sicherstellen, dass das Feld auf dem Device liegt
    dev_force_field = _to_backend(backend, force_field)

    kernel = vector_field_force_kernel!(backend)
    kernel(
        state.momentum.x,
        state.momentum.y,
        state.momentum.z,
        state.mass,
        dev_force_field,
        T(dt);
        ndrange=size(state.mass)
    )

    KernelAbstractions.synchronize(backend)
end