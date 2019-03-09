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
either 
linear programs,
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

`cpsr` can be used in combination with [`cvxpy`](https://www.cvxpy.org),
via the `cpsr.cvxpy_solve` method. An example follows.

```python
import numpy as np
import cvxpy as cp
import cpsr


np.random.seed(1)
 
n = 4

X = cp.Variable((n,n))

problem = cp.Problem(objective = cp.Minimize(cp.norm(X - np.random.randn(n, n))), 
                     constraints = [X @ np.random.randn(n) == np.random.randn(n)])

problem.solve(solver='SCS', verbose=False)
print('constraint violation with default solver')
print(problem.constraints[0].violation())

cpsr.cvxpy_solve(problem, presolve = True, verbose=False)
print('constraint violation with CPSR')
print(problem.constraints[0].violation())

cpsr.cvxpy_solve(problem, verbose=False)
print('and running CPSR again')
print(problem.constraints[0].violation())
```

It has the following output.

```
constraint violation with default solver
[1.13710559e-06 2.39427364e-06 4.49123915e-06 2.87486420e-07]
constraint violation with CPSR
[5.49834345e-11 2.64619437e-11 3.22242233e-11 2.61812794e-11]
and running CPSR again
[3.66026653e-15 1.99840144e-15 2.77555756e-17 2.44249065e-15]
```
