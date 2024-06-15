## Julia ~ 850μs 
Lastly, we look at [Julia](https://docs.julialang.org/en/v1/), another member of the [LLVM](https://en.wikipedia.org/wiki/LLVM) family.


```julia
function random_spin_field(N:: Integer, M:: Integer)::Matrix{Int8}
    return rand([-1, 1], N, M)
end
```


    random_spin_field (generic function with 1 method)



```julia
function ising_step(field::Matrix{Int8}, beta::Float32, func)::Matrix{Int8}
    N, M = size(field)
    for n_offset in 1:2
        for m_offset in 1:2
            for n in n_offset:2:N-1
                for m in m_offset:2:M-1
                    func(field, n, m, beta)
                end
            end
        end
    end
    return field
end
```


    ising_step (generic function with 1 method)


Julia translates pretty closely from Python, just take note of 1-indexed arrays instead of 0-indexed arrays.


```julia
function _ising_step(field::Matrix{Int8}, n::Integer, m::Integer, beta::Float32)
    total = 0
    N, M = size(field)
    for i in n-1:n+1
        for j in m-1:m+1
            if i == n && j == m
                continue
            end
            # Convert to 0-indexing
            i -= 1
            j -= 1
            # Take the remainder and convert back to 1-indexing.
            total += field[abs(i % N) + 1, abs(j % M) + 1]
        end
    end
    dE = 2 * field[n, m] * total
    if dE <= 0
        field[n, m] *= -1
    elseif exp(-dE * beta) > rand()
        field[n, m] *= -1
    end
end
```


    _ising_step (generic function with 1 method)



```julia
N, M = 200, 200
field = random_spin_field(N, M)
ising_step(field, 0.04f0, _ising_step)
println(size(field))
```

    (200, 200)



```julia
using BenchmarkTools
@btime ising_step(field, 0.04f0, _ising_step)
println("")
```

      853.750 μs (0 allocations: 0 bytes)
    


Which almost runs as fast as Cython


```julia
run(`jupyter nbconvert --to markdown ising_model_speed_3.ipynb`)
```
