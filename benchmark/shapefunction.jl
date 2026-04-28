using BenchmarkTools
using StaticArrays



@fastmath @inline function shapefunction_original(natural_coords::SVector{3, T}) where {T}
    Nx = max(one(T) - abs(natural_coords[1]), zero(T))
    Ny = max(one(T) - abs(natural_coords[2]), zero(T))
    Nz = max(one(T) - abs(natural_coords[3]), zero(T))
    
    return Nx * Ny * Nz
end


@fastmath @inline function shapefunction_oneliner(natural_coords::SVector{3, T}) where {T}
    return prod(max.(one(T) .- abs.(natural_coords), zero(T)))
end

@fastmath @inline function shapefunction_break(natural_coords::SVector{3, T}) where {T}
    Nx = max(one(T) - abs(natural_coords[1]), zero(T))
    Nx == zero(T) && return zero(T) 
    
    Ny = max(one(T) - abs(natural_coords[2]), zero(T))
    Ny == zero(T) && return zero(T) 
    
    Nz = max(one(T) - abs(natural_coords[3]), zero(T))

    return Nx * Ny * Nz
end


#compile all
vec = SVector{3, Float64}(randn(3))
shapefunction_original(vec)
shapefunction_oneliner(vec)
shapefunction_break(vec)

const test_points = [SArray{Tuple{3}}(randn(3)) for _ in 1:1000] .* 4/3

function benchmark_loop(points, func)
    total = 0.0
    for p in points
        total += func(p)
    end
    return total
end

println("Loop-Benchmark (1000 Punkte):")
print("Original:   ")
@btime benchmark_loop($test_points, shapefunction_original)
#Result: 761.150 ns (0 allocations: 0 bytes)
# ==> This one is chosen

print("Inbounds:   ")
@btime benchmark_loop($test_points, shapefunction_oneliner)
#Result: 767.444 ns (0 allocations: 0 bytes)

print("Early-Exit: ")
@btime benchmark_loop($test_points, shapefunction_break) 