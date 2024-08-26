A post covering how to complete matrix inversions in [PyTorch](https://pytorch.org/) using [BLAS and LAPACK operations](https://pytorch.org/docs/stable/torch.html#blas-and-lapack-operations). This is particularly useful in Statistics for inverting [covariance matrices](https://en.wikipedia.org/wiki/Covariance_matrix) to form statistical estimators. I came across this recently during [a code review](https://github.com/pytorch/botorch/pull/2474) and wanted to collect my thoughts on the topic.

## Matrix Inverse
First, I review how to compute a matrix-inverse product exactly the way they are presented in statistical textbooks.

$$
\begin{align*}
Ax &= b \\
x &= A^{-1}b \\
\end{align*}
$$

Where $A \in \mathbb R ^{n \times n}$ is a [positive semi-definite (PSD)](https://en.wikipedia.org/wiki/Definite_matrix#Definitions_for_real_matrices) matrix:

$$
\begin{align*}
x^T A x &\geq 0
\end{align*}
$$

For all non-zero $x \in \mathbb R^n$. We can solve this in PyTorch by using [`torch.linalg.inv`](https://pytorch.org/docs/stable/generated/torch.linalg.inv.html):


```python
import torch

# Set the seed for reproducibility
torch.manual_seed(1)
# Use double precision
torch.set_default_dtype(torch.float64)

a = torch.randn(3, 3)
b = torch.randn(3, 2)
# Create a PSD matrix
A = a @ a.T + torch.eye(3) * 1e-3
x = torch.linalg.inv(A) @ b
x_original = x
x
```




    tensor([[-3.2998, -3.1372],
            [-0.6995, -0.0583],
            [-1.3701, -0.7586]])



The problem with this is that it can become numerically instable for [poorly conditioned matrices](https://en.wikipedia.org/wiki/Condition_number#Matrices). 

> In general, we want to [use matrix decompositions and avoid inverting matrices](https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/) as shown in [this blog post](https://civilstat.com/2015/07/dont-invert-that-matrix-why-and-how/). 
{: .prompt-warning }

## Cholesky Decomposition

We can avoid a matrix inverse by considering the [cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) of $A$ (giving us a lower triangular matrix $L$):

$$
\begin{align*}
A &= LL^T \\
LL^T x &= b \\
L^T x &= c \\
Lc &= b
\end{align*}
$$

$$
\begin{align*}
\text{Forward solve } Lc &= b \text{ for $c$} \\
\text{Backwards solve } L^Tx &= c \text{ for $x$} \\
\end{align*}
$$

Which can be solved efficiently using [forwards-backwards substitution](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution). In PyTorch, we can use [torch.cholesky](https://pytorch.org/docs/stable/generated/torch.cholesky.html) and [torch.cholesky_solve](https://pytorch.org/docs/stable/generated/torch.cholesky_solve.html):


```python
L = torch.linalg.cholesky(A)
x = torch.cholesky_solve(b, L)
x
```




    tensor([[-3.2998, -3.1372],
            [-0.6995, -0.0583],
            [-1.3701, -0.7586]])




```python
torch.dist(x, x_original)
```




    tensor(2.6879e-15)



Importantly, this can be accomplished through matrix multiplication and forwards-backwards algorithm without taking any matrix inverse (see [this comment](https://github.com/pytorch/pytorch/issues/77166#issuecomment-1122996050) for a description).

## LU Decomposition

We can also do this with a [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition) where $L$ is a lower triangular matrix and $U$ is an upper triangular matrix:

$$
\begin{align*}
A &= LU \\
LU x &= b \\
\text{Set } U x &= c \\
Lc &= b
\end{align*}
$$

$$
\begin{align*}
\text{Forward solve } Lc &= b \text{ for $c$} \\
\text{Backwards solve } Ux &= c \text{ for $x$} \\
\end{align*}
$$

In PyTorch, we can use [`torch.linalg.lu_factor`](https://pytorch.org/docs/stable/generated/torch.linalg.lu_factor.html#torch.linalg.lu_factor) and [`torch.linalg.lu_solve`](https://pytorch.org/docs/stable/generated/torch.linalg.lu_solve.html#torch.linalg.lu_solve):


```python
LU, pivots = torch.linalg.lu_factor(A)
x = torch.linalg.lu_solve(LU, pivots, b)
x
```




    tensor([[-3.2998, -3.1372],
            [-0.6995, -0.0583],
            [-1.3701, -0.7586]])




```python
torch.dist(x, x_original)
```




    tensor(8.5664e-16)



## QR Decomposition
We can also use the [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) where $Q$ is an orthogonal matrix and $R$ is upper right triangular matrix:

$$
\begin{align*}
A &= QR \\
QR x &= b \\
\text{Set } R x &= c \\
Qc &= b \\
c &= Q^T b
\end{align*}
$$

$$
\begin{align*}
\text{Backwards solve } Rx &= c \text{ for $x$} \\
\end{align*}
$$

In PyTorch, we can use [`torch.linalg.qr`](https://pytorch.org/docs/stable/generated/torch.qr.html#torch.qr) and [`torch.linalg.solve_triangular`](https://pytorch.org/docs/stable/generated/torch.linalg.solve_triangular.html#torch.linalg.solve_triangular):


```python
Q, R = torch.linalg.qr(A)
c = Q.T @ b
x = torch.linalg.solve_triangular(R, c, upper=True)
x
```




    tensor([[-3.2998, -3.1372],
            [-0.6995, -0.0583],
            [-1.3701, -0.7586]])




```python
torch.dist(x, x_original)
```




    tensor(2.1020e-15)



## SVD
We can also use the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition#Calculating_the_SVD) where $U$ is an orthogonal matrix, $S$ is a diagonal matrix, and $V$ is a orthogonal matrix (in the special case that $A$ is a real square matrix):

$$
\begin{align*}
A &= USV^T \\
USV^Tx &= b \\
x &= (VS^{-1}U^T)b
\end{align*}
$$

> Note, we do take $S^{-1}$, but since this is a diagonal matrix, the inverse can be computed analytically by simply inverting each diagonal entry.
{: .prompt-tip }

 In PyTorch, we can use [`torch.linalg.svd`](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html#torch.linalg.svd):


```python
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
x = Vh.T @ torch.diag(1 / S) @ U.T @ b
x
```




    tensor([[-3.2998, -3.1372],
            [-0.6995, -0.0583],
            [-1.3701, -0.7586]])




```python
torch.dist(x, x_original)
```




    tensor(1.6614e-14)



## Blackbox Matrix-Matrix Multiplication (BBMM)
Since matrix inversion is especially relevant to [Gaussian Processes](https://en.wikipedia.org/wiki/Gaussian_process), the library [GPyTorch](https://gpytorch.ai/) has implemented a [Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration](https://arxiv.org/abs/1809.11165) library. Importantly, it lowers the cost of the above approaches from $O(n^3)$ to $O(n^2)$ and allows routines to be used on GPU architectures. GPyTorch uses [LinearOperator](https://github.com/cornellius-gp/linear_operator) which is useful for exploiting specific matrix structure:


```python
from linear_operator.operators import DiagLinearOperator, LowRankRootLinearOperator
C = torch.randn(1000, 20)
d = torch.ones(1000) * 1e-9
b = torch.randn(1000)
A = LowRankRootLinearOperator(C) + DiagLinearOperator(d)
_ = torch.linalg.solve(A, b)
```
