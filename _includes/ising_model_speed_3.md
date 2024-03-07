## Julia ~ 3.2ms
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


### Naive ~ 3.2ms
Julia translates pretty closely from Python, just take note of 1-indexed arrays instead of 0-indexed arrays.


```julia
function _ising_step(field::Matrix{Int8}, n::Integer, m::Integer, beta::Float32)
    total = 0
    N, M = size(field)

    # 1-indexed arrays cannot use the % trick from before.
    nm1 = n - 1 == 0 ? N : n
    np1 = n + 1 == N ? 1 : n
    mm1 = m - 1 == 0 ? M : m
    mp1 = m + 1 == M ? 1 : m

    for i in [nm1, n, np1]
        for j in [mm1, m, mp1]
            if i == n && j == m
                continue
            end
            total += field[i, j]
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

      3.282 ms (158404 allocations: 12.09 MiB)
    


### Unrolled ~ 1.3ms
We can also include the unrolled version from before.


```julia
function _ising_step_unrolled(field::Matrix{Int8}, n::Integer, m::Integer, beta::Float32)
    total = 0
    N, M = size(field)
    nm1 = n - 1 == 0 ? N : n
    np1 = n + 1 == N ? 1 : n
    mm1 = m - 1 == 0 ? M : m
    mp1 = m + 1 == M ? 1 : m
    dE = (
        2
        * field[n, m]
        * (
            field[nm1, mm1]
            + field[nm1, m]
            + field[nm1, mp1]
            + field[n, mm1]
            + field[n, mp1]
            + field[np1, mm1]
            + field[np1, m]
            + field[np1, mp1]
        )
    )
    if dE <= 0
        field[n, m] *= -1
    elseif exp(-dE * beta) > rand()
        field[n, m] *= -1
    end
end
```


    _ising_step_unrolled (generic function with 1 method)



```julia
@btime ising_step(field, 0.04f0, _ising_step_unrolled)
println("")
```

      1.302 ms (0 allocations: 0 bytes)
    


Which runs around the speed of Mojo when using for-loops and around the speed of Numba when using unrolled.


```julia
run(`jupyter nbconvert --to markdown ising_model_speed_3.ipynb`)
```
