# `cpsr`: Cone Program Solution Refinement

`cpsr` is a [Python]([https://www.python.org) library 
for the iterative improvement, or refinement,
of a primal-dual solution,
or a certificate of unboundedness or infeasibility,
of a cone program. 

Given an approximate solution (or certificate), 
meaning one for which the optimality 
conditions don't hold exactly, 
`cpsr` produces a new solution for which 
the norm of the violations of the primal and dual constraints, 
and the duality gap, is smaller. 

It does so by computing the gradient 
of the operator 𝒩 (z) ∈ 𝗥^(n), 
where z ∈ 𝗥^(n) is a primal-dual approximate solution,
and 𝒩 (z) = 0 if and only if z in an *exact* primal-dual solution,
or certificate, meaning one for which the optimality conditions
are satisfied within machine precision.

It currently supports cone programs that are
either linear programs,
second-order cone programs, 
exponential programs, 
semidefinite programs,
and any combination. 

To install, execute in a terminal:

```
pip install cpsr
```

`cpsr` depends on [`numpy`](http://www.numpy.org) for vector arithmetics, 
[`scipy`](https://www.scipy.org) for sparse linear algebra,
and [`numba`](https://numba.pydata.org) for just-in-time code compilation.

A detailed description of the algorithm used is provided
in [the accompanying paper](http://stanford.edu/~boyd/papers/cone_prog_refine.html).

#### `cvxpy` interface

`cpsr` can be used in combination with [`cvxpy`](https://www.cvxpy.org)
to 


a  problem. We currently
offer

In this example, the problem is first solved with the default settings of
`cvxpy`, then with the `cvxpy_solve` method of `cpsr`, and then again
which currently 
runs [`scs`](https://github.com/cvxgrp/scs)

```
import numpy as np
import cvxpy as cvx
import cpsr


x = cvx.Variable(10)
np.random.seed(1)
A = np.random.randn(5,10)
b = np.random.randn(5)

problem = cvx.Problem(cvx.Minimize(cvx.norm(x)),  [A @ x >= b])

problem.solve()
error_one = np.minimum( A @ x.value - b, 0.)

cpsr.cvxpy_solve(problem, presolve = True)
error_two = np.minimum( A @ x.value - b, 0.)

cpsr.cvxpy_solve(problem, presolve = False, warm_start = True)
error_three = np.minimum( A @ x.value - b, 0.)

print(error_one)
print(error_two)
print(error_three)
```
