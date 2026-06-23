# ---------------------------------------------------------------------------- #
#                               Device Management                              #
# ---------------------------------------------------------------------------- #
function _to_backend(backend, arr::AbstractArray{T, N}) where {T, N}
    dest = KernelAbstractions.allocate(backend, T, size(arr))
    copyto!(dest, arr)
    return dest
end

_to_backend(::CPU, arr::AbstractArray) = arr    # Do nothing for CPU backend, just return the array as is