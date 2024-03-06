## Mojo ~ 3ms
[Mojo](https://docs.modular.com/mojo/) is a relatively newcomer to the scientific computing scene, but is quickly gaining popularity.

> Mojo codeblocks display as Python since markdown does not yet support Mojo highlighting.
{: .prompt-info }

First, we will need to import a variety of items from the standard library.


```python
from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from random import rand
from math import exp
from benchmark import Report
import benchmark

alias data_type = DType.float32
```

Now, we can rewrite our functions in Mojo. First, we start with the `random_spin_field`:


```python
fn random_spin_field(N: Int, M: Int) -> Tensor[data_type]:
    var t = rand[data_type](N, M)
    for i in range(N):
        for j in range(M):
            if t[Index(i, j)] < 0.5:
                t[Index(i, j)] = -1
            else:
                t[Index(i, j)] = 1
    return t
```

Next, the internal `_ising_update` which takes the summation over the neighbors:


```python
fn _ising_update(inout field: Tensor[data_type], n: Int, m: Int, beta: Float32) -> None:
    var total = SIMD[data_type, 1]()
    var shape = field.shape()
    var N = shape[0]
    var M = shape[1]
    for i in range(n - 1, n + 2):
        for j in range(m - 1, m + 2):
            if i == n and j == m:
                continue
            total += field[Index(i % N, j % M)]
    var dE = 2 * field[Index(n, m)] * total
    if dE <= 0:
        field[Index(n, m)] = -field[Index(n, m)]
    elif exp(-dE * beta) > rand[data_type](1)[Index(0)]:
        field[Index(n, m)] = -field[Index(n, m)]
```

Lastly, we can define the `ising_step`:


```python
fn ising_step(inout field: Tensor[data_type], beta: Float32 = 0.4) -> None:
    var shape = field.shape()
    var N = shape[0]
    var M = shape[1]
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    _ising_update(field=field, n=n, m=m, beta=beta)
```

We can define a small benchmark function.


```python
@always_inline
fn bench() -> Report:
    var N = 200
    var M = 200
    var field = random_spin_field(N, M)

    @always_inline
    @parameter
    fn ising_step_fn():
        ising_step(field=field)

    return benchmark.run[ising_step_fn](max_runtime_secs=10)

var report = bench()
```


```python
# Print a report in Milliseconds
report.print("ms")
```

    ---------------------
    Benchmark Report (ms)
    ---------------------
    Mean: 2.989141800246609
    Total: 2424.194
    Iters: 811
    Warmup Mean: 2.5299999999999998
    Warmup Total: 5.0599999999999996
    Warmup Iters: 2
    Fastest Mean: 2.989141800246609
    Slowest Mean: 2.989141800246609
    


We see that Mojo runs a little bit slower than Numba without optimization.


```python
%%python
# A little magic to automatically write my blog :)
import subprocess

subprocess.run(["jupyter", "nbconvert", "--to", "markdown", "ising_model_speed_2.ipynb"])
subprocess.run("sed -i'' -e 's/```python/```python/g' ising_model_speed_2.md", shell=True)
```
