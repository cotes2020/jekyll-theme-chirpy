## Python ~ 140ms
Without changing [@Jake VanderPlas](https://github.com/jakevdp)'s code, lets rerun a vanilla Python for-loop and see how this speed compares with the speed from 2017 in the [original blogpost](https://jakevdp.github.io/blog/2017/12/11/live-coding-cython-ising-model/).


```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, Markdown

np.random.seed(1)
```


```python
# Plotting helper functions.
def render_plotly_html(fig: go.Figure) -> None:
    """Display a Plotly figure in markdown with HTML."""
    # Ensure frame is square
    display(
        Markdown(
            fig.to_html(
                include_plotlyjs="cdn",
            )
        )
    )
```


```python
# Functions from Jake VanderPlas's blog post.
def random_spin_field(N, M):
    return np.random.choice([-1, 1], size=(N, M))


# Add an `update_fn` arg.
def ising_step(field, beta=0.4):
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    _ising_update(field, n, m, beta)
    return field


def _ising_update(field, n, m, beta):
    total = 0
    N, M = field.shape
    for i in range(n - 1, n + 2):
        for j in range(m - 1, m + 2):
            if i == n and j == m:
                continue
            total += field[i % N, j % M]
    dE = 2 * field[n, m] * total
    if dE <= 0:
        field[n, m] *= -1
    elif np.exp(-dE * beta) > np.random.rand():
        field[n, m] *= -1
```


```python
N, M = 200, 200
field = random_spin_field(N, M)
```


```python
%timeit ising_step(field)
```

    138 ms ± 1.92 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


It appears Python has some (2x) inherent speedup over time.

## Cython ~ 580 µs
And similarly, for the Cython code.


```python
%load_ext Cython
```


```cython
%%cython

cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_ising_step(np.int64_t[:, :] field, float beta=0.4):
    cdef int N = field.shape[0]
    cdef int M = field.shape[1]
    cdef int n_offset, m_offset, n, m
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    _cy_ising_update(field, n, m, beta)
    return np.array(field)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _cy_ising_update(np.int64_t[:, :] field, int n, int m, float beta):
    cdef int total = 0
    cdef int N = field.shape[0]
    cdef int M = field.shape[1]
    cdef int i, j
    for i in range(n-1, n+2):
        for j in range(m-1, m+2):
            if i == n and j == m:
                continue
            total += field[i % N, j % M]
    cdef float dE = 2 * field[n, m] * total
    if dE <= 0:
        field[n, m] *= -1
    elif exp(-dE * beta) * RAND_MAX > rand():
        field[n, m] *= -1
```


```python
%timeit cy_ising_step(field)
```

    583 µs ± 5.44 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


Cython also has become faster (4x) since 2017.

## Numba ~ 1.1ms
[@Jake VanderPlas](https://github.com/jakevdp) also mentions [another blog](https://matthewrocklin.com/blog/work/2015/02/28/Ising) that used [Numba](https://numba.readthedocs.io/en/stable/index.html) to speed things up. Lets stick with our original implementation and see how far we can get.

### Naive @njit ~ 1.1ms
To use basic Numba, we just need to rewrite our update function and add the [`@njit`](https://numba.readthedocs.io/en/stable/user/jit.html) decorator to the function.


```python
from numba import njit
```


```python
@njit
def numba_ising_step(field, beta=0.4):
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    total = 0
                    for i in range(n - 1, n + 2):
                        for j in range(m - 1, m + 2):
                            if i == n and j == m:
                                continue
                            total += field[i % N, j % M]
                    dE = 2 * field[n, m] * total
                    if dE <= 0:
                        field[n, m] *= -1
                    elif np.exp(-dE * beta) > np.random.rand():
                        field[n, m] *= -1
    return field
```


```python
# Precompile numba_ising_step
numba_ising_step(field)
# Time the code
%timeit numba_ising_step(field)
```

    1.09 ms ± 7.63 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


Although faster than Python, it is still much slower than Cython.

### @njit(parallel=True) ~ 650µs
Before trying to speed this up, lets first understand a common exploit of the Ising Model. First, lets extract our outer for-loops and simply keep record of the `n`'s, `m`s, and offset.


```python
ns = []
ms = []
offsets = []
for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, 10, 2):
                  for m in range(m_offset, 10, 2):
                        ns.append(n)
                        ms.append(m)
                        offsets.append((n_offset, m_offset))
```

Now, lets plot them colored by their offset. I include the index which is the order of our traversal.


```python
fig = px.scatter(
    pd.DataFrame({"n": ns, "m": ms, "offset": offsets}).reset_index(),
    x="n",
    y="m",
    color="offset",
    title="Order of n vs. m",
    text="index",
)
fig.update_traces(textposition="bottom right")
render_plotly_html(fig)
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>                <div id="eb3fad7d-c98f-45fd-87b5-4f95ce3cba3e" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("eb3fad7d-c98f-45fd-87b5-4f95ce3cba3e")) {                    Plotly.newPlot(                        "eb3fad7d-c98f-45fd-87b5-4f95ce3cba3e",                        [{"hovertemplate":"offset=(0, 0)\u003cbr\u003en=%{x}\u003cbr\u003em=%{y}\u003cbr\u003eindex=%{text}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"(0, 0)","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers+text","name":"(0, 0)","orientation":"v","showlegend":true,"text":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0],"x":[0,0,0,0,0,2,2,2,2,2,4,4,4,4,4,6,6,6,6,6,8,8,8,8,8],"xaxis":"x","y":[0,2,4,6,8,0,2,4,6,8,0,2,4,6,8,0,2,4,6,8,0,2,4,6,8],"yaxis":"y","type":"scatter","textposition":"bottom right"},{"hovertemplate":"offset=(0, 1)\u003cbr\u003en=%{x}\u003cbr\u003em=%{y}\u003cbr\u003eindex=%{text}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"(0, 1)","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers+text","name":"(0, 1)","orientation":"v","showlegend":true,"text":[25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0],"x":[0,0,0,0,0,2,2,2,2,2,4,4,4,4,4,6,6,6,6,6,8,8,8,8,8],"xaxis":"x","y":[1,3,5,7,9,1,3,5,7,9,1,3,5,7,9,1,3,5,7,9,1,3,5,7,9],"yaxis":"y","type":"scatter","textposition":"bottom right"},{"hovertemplate":"offset=(1, 0)\u003cbr\u003en=%{x}\u003cbr\u003em=%{y}\u003cbr\u003eindex=%{text}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"(1, 0)","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers+text","name":"(1, 0)","orientation":"v","showlegend":true,"text":[50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0],"x":[1,1,1,1,1,3,3,3,3,3,5,5,5,5,5,7,7,7,7,7,9,9,9,9,9],"xaxis":"x","y":[0,2,4,6,8,0,2,4,6,8,0,2,4,6,8,0,2,4,6,8,0,2,4,6,8],"yaxis":"y","type":"scatter","textposition":"bottom right"},{"hovertemplate":"offset=(1, 1)\u003cbr\u003en=%{x}\u003cbr\u003em=%{y}\u003cbr\u003eindex=%{text}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"(1, 1)","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers+text","name":"(1, 1)","orientation":"v","showlegend":true,"text":[75.0,76.0,77.0,78.0,79.0,80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,96.0,97.0,98.0,99.0],"x":[1,1,1,1,1,3,3,3,3,3,5,5,5,5,5,7,7,7,7,7,9,9,9,9,9],"xaxis":"x","y":[1,3,5,7,9,1,3,5,7,9,1,3,5,7,9,1,3,5,7,9,1,3,5,7,9],"yaxis":"y","type":"scatter","textposition":"bottom right"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"n"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"m"}},"legend":{"title":{"text":"offset"},"tracegroupgap":0},"title":{"text":"Order of n vs. m"}},                        {"responsive": true}                    )                };                            </script>        </div>
</body>
</html>



```python
fig.show()
```



If you click on any one of the offsets, you'll notice that when that offset disappears, all of its neighbors stay intact. Because the markov blanket of each cell in the Ising Model are the immediate neighbors, this means these update steps are independent, and we can do them in parallel using Numba's [`parallel=True`](https://numba.readthedocs.io/en/stable/user/parallel.html).


```python
from numba import prange
```


```python
@njit(parallel=True)
def numba_parallel_ising_step(field, beta=0.4):
    N, M = field.shape

    for n_offset in range(2):
        for m_offset in range(2):
            ns = np.arange(n_offset, N, 2)
            for n in prange(len(ns)):
                n = ns[n]
                ms = np.arange(m_offset, M, 2)
                for m in prange(len(ms)):
                    m = ms[m]
                    total = 0
                    for i in range(n - 1, n + 2):
                        for j in range(m - 1, m + 2):
                            if i == n and j == m:
                                continue
                            total += field[i % N, j % M]
                    dE = 2 * field[n, m] * total
                    if dE <= 0:
                        field[n, m] *= -1
                    elif np.exp(-dE * beta) > np.random.rand():
                        field[n, m] *= -1
    return field
```


```python
numba_parallel_ising_step(field)
%timeit numba_parallel_ising_step(field)
```

    645 µs ± 44 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


`prange` also has some ability to detect parallel-eligible code itself. For example, the iteration over all the neighbors, can happen in parallel.


```python
numba_parallel_ising_step.parallel_diagnostics(level=1)
```

     
    ================================================================================
     Parallel Accelerator Optimizing:  Function numba_parallel_ising_step, 
    /var/folders/1j/7lmy8byj4sv720g4hf67kkmm0000gn/T/ipykernel_2188/3547723262.py 
    (1)  
    ================================================================================
    
    
    Parallel loop listing for  Function numba_parallel_ising_step, /var/folders/1j/7lmy8byj4sv720g4hf67kkmm0000gn/T/ipykernel_2188/3547723262.py (1) 
    -------------------------------------------------------------------|loop #ID
    @njit(parallel=True)                                               | 
    def numba_parallel_ising_step(field, beta=0.4):                    | 
        N, M = field.shape                                             | 
                                                                       | 
        for n_offset in range(2):                                      | 
            for m_offset in range(2):                                  | 
                ns = np.arange(n_offset, N, 2)                         | 
                for n in prange(len(ns)):------------------------------| #3
                    n = ns[n]                                          | 
                    ms = np.arange(m_offset, M, 2)                     | 
                    for m in prange(len(ms)):--------------------------| #2
                        m = ms[m]                                      | 
                        total = 0                                      | 
                        for i in range(n - 1, n + 2):                  | 
                            for j in range(m - 1, m + 2):              | 
                                if i == n and j == m:                  | 
                                    continue                           | 
                                total += field[i % N, j % M]           | 
                        dE = 2 * field[n, m] * total                   | 
                        if dE <= 0:                                    | 
                            field[n, m] *= -1                          | 
                        elif np.exp(-dE * beta) > np.random.rand():    | 
                            field[n, m] *= -1                          | 
        return field                                                   | 
    ------------------------------ After Optimisation ------------------------------
    Parallel region 0:
    +--3 (parallel)
       +--1 (serial, fused with loop(s): 2)
       +--2 (serial)
    
    
    Parallel region 1:
    +--0 (parallel)
       +--1 (serial)
    
    
     
    Parallel region 0 (loop #3) had 1 loop(s) fused and 2 loop(s) serialized as part
     of the larger parallel loop (#3).
     
    Parallel region 1 (loop #0) had 0 loop(s) fused and 1 loop(s) serialized as part
     of the larger parallel loop (#0).
    --------------------------------------------------------------------------------
    --------------------------------------------------------------------------------
     


With `parallel=True`, we are now in the neighborhood of Cython.

### Unrolling for-loops ~ 600µs
For small for-loops (like iterating over the neighbors of an atom), we can use a trick called [unrolling the for-loop](https://en.wikipedia.org/wiki/Loop_unrolling). Basically, we rewrite the for-loop as an explicit calculation. I have seen this used successively in Micropython as well.


```python
@njit(parallel=True)
def numba_parallel_unrolled_ising_step(field, beta=0.4):
    N, M = field.shape

    for n_offset in range(2):
        for m_offset in range(2):
            ns = np.arange(n_offset, N, 2)
            for n in prange(len(ns)):
                n = ns[n]
                ms = np.arange(m_offset, M, 2)
                for m in prange(len(ms)):
                    m = ms[m]
                    dE = (
                        2
                        * field[n, m]
                        # Unrolled for-loop over 8 neighbors.
                        * (
                            field[(n - 1) % N, (m - 1) % M]
                            + field[(n - 1) % N, m]
                            + field[(n - 1) % N, (m + 1) % M]
                            + field[n, (m - 1) % M]
                            + field[n, (m + 1) % M]
                            + field[(n + 1) % N, (m - 1) % M]
                            + field[(n + 1) % N, m]
                            + field[(n + 1) % N, (m + 1) % M]
                        )
                    )
                    if dE <= 0:
                        field[n, m] *= -1
                    elif np.exp(-dE * beta) > np.random.rand():
                        field[n, m] *= -1
    return field
```


```python
numba_parallel_unrolled_ising_step(field)
%timeit numba_parallel_unrolled_ising_step(field)
```

    593 µs ± 33.2 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


Nice! Faster, but not beating Cython conclusively.


```python
# A little magic to automatically write my blog :)
import subprocess

subprocess.run(["jupyter", "nbconvert", "--to", "markdown", "ising_model_speed.ipynb"])
```
