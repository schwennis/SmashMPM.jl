# ---------------------------------------------------------------------------- #
#                               Device Management                              #
# ---------------------------------------------------------------------------- #
function _to_backend(backend, arr::AbstractArray{T, N}) where {T, N}
    dest = KernelAbstractions.allocate(backend, T, size(arr))
    copyto!(dest, arr)
    return dest
end

_to_backend(::CPU, arr::AbstractArray) = arr    # Do nothing for CPU backend, just return the array as is


# ---------------------------------------------------------------------------- #
#                         atomic_max!, backend specific                        #
# ---------------------------------------------------------------------------- #
# Default (GPU etc.): native atomic max
@inline function atomic_max!(A::AbstractArray, i, j, k, val)
    Atomix.@atomic :monotonic A[i, j, k] max val
    return nothing
end

# CPU: UnsafeAtomics has no float `max`, so use a CAS loop
@inline function atomic_max!(A::Array{T}, i, j, k, val) where {T}
    v = T(val)
    old = Atomix.@atomic :monotonic A[i, j, k]
    while v > old
        res = Atomix.@atomicreplace :monotonic :monotonic A[i, j, k] old => v
        res.success && break
        old = res.old
    end
    return nothing
end