# `cpsr`: Cone Program Solution Refinement

`cpsr` is a pure-Python library for the refinement of solutions of 
primal-dual cone programs. It currently supports linear programs,
second-order cone programs, exponential programs, semidefinite programs,
and any combination. To install, execute in a terminal:

```
pip install cpsr
```

`cpsr` depends on [`numpy`](http://www.numpy.org) for vector arithmetic, 
[`scipy`](https://www.scipy.org) for sparse linear algebra,
and [`numba`](https://numba.pydata.org) for just-in-time code compilation.

A detailed description of the algorithm used is provided
in the accompanying 
[paper](http://stanford.edu/~boyd/papers/cone_prog_refine.html),
which you can cite if you find the program useful.

#### `cvxpy` interface

A simple way to use CPSR is by refining the solution of
a [`cvxpy`](https://www.cvxpy.org) problem, *e.g.*,

```
import numpy as np
import cvxpy as cvx
import cpsr


x = cvx.Variable(10)
A = np.random.randn(5,10)
b = np.random.randn(5)

problem = cvx.Problem(cvx.Minimize(cvx.norm(x)),  [A @ x >= b])
cpsr.cvxpy_solve(problem, presolve = True, verbose = True, iters = 2, lsqr_iters = 30)

print(np.minimum( A @ x.value - b, 0.))

cpsr.cvxpy_solve(problem, presolve = False, warm_start = True, iters = 2, lsqr_iters = 30)

print(np.minimum( A @ x.value - b, 0.))

```
