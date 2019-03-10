# `cpsr`: Cone Program Solution Refinement

`cpsr` is a [Python](https://www.python.org) library 
for the iterative improvement, or refinement,
of a primal-dual solution,
or a certificate of unboundedness or infeasibility,
of a cone (or [convex](http://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)) program. 
It operates by differentiating the conic optimality conditions,
and so it can also be used for *calculus* of conic programs.

### Refinement

Given an approximate solution (or certificate), 
meaning one for which the optimality 
conditions don't hold exactly, 
`cpsr` produces a new solution for which 
the norm of the violations of the primal and dual constraints, 
and the duality gap, is smaller. 

It does so by locally linearizing
the operator 𝒩 (z) ∈ 𝗥^(n), 
the concatenation of the violations of the 
primal and dual constraints, and the duality gap,
for any approximate primal-dual solution (or certificate) z ∈ 𝗥^(n).
So, 𝒩 (z) = 0 if and only if z is an exact primal-dual solution
or certificate, meaning one for which the optimality conditions
are satisfied within machine precision. 

`cpsr` is a matrix-free solver, meaning that it does not store or
invert the derivative matrix of 𝒩 (z). This allows it to scale
to very large problems. Essentially, if you are able to load the problem
data in memory, then `cpsr` can solve it, with O(n) memory requirement.

It uses [LSQR](http://web.stanford.edu/group/SOL/software/lsqr/),
an iterative linear system solver, to approximately solve the system
that locally approximates the conic optimality conditions. 
The number of LSQR iterations is chosen by the user (by default, for small problems, 30),
as is the number of `cpsr` iterations (by default, for small problems, 2). 

It can currently solve cone programs whose cone constraints are products of 
the [zero cone](https://en.wikipedia.org/wiki/System_of_linear_equations),
[the non-negative orhant](https://en.wikipedia.org/wiki/Linear_programming),
and any number of [second-order cones](https://en.wikipedia.org/wiki/Second-order_cone_programming), 
[exponential cones](https://yalmip.github.io/tutorial/exponentialcone/), 
and [semidefinite cones](https://en.wikipedia.org/wiki/Semidefinite_programming).

A much more detailed description of the algorithm used is provided
in [the accompanying paper](http://stanford.edu/~boyd/papers/cone_prog_refine.html).

I show the experiments described in the paper in the
[Jupyter](https://jupyter.org)
notebook
[examples/experiments.ipynb](examples/experiments.ipynb).


### Installation
To install, execute in a terminal:

```
pip install cpsr
```

`cpsr` depends on [`numpy`](http://www.numpy.org) for vector arithmetics, 
[`scipy`](https://www.scipy.org) for sparse linear algebra,
and [`numba`](https://numba.pydata.org) for just-in-time code compilation.
It currently runs on a single thread, on CPU. I plan to support 
multi-threading and GPUs, 
and rewrite the algorithmic part of the code in C. I also plan to
release interfaces to other scientific programming languages.


### `cvxpy` interface

`cpsr` can be used in combination with [`cvxpy`](https://www.cvxpy.org),
via the `cpsr.cvxpy_solve` method. An example follows.

```python
import numpy as np
import cvxpy as cp
import cpsr

n = 5
np.random.seed(1)
X = cp.Variable((n,n))

problem = cp.Problem(objective = cp.Minimize(cp.norm(X - np.random.randn(n, n))), 
                     constraints = [X @ np.random.randn(n) == np.random.randn(n)])

cpsr.cvxpy_solve(problem, presolve=True, verbose=False)
print('norm of the constraint error, solving with SCS and then refining with CPSR: %.2e' % 
      np.linalg.norm(problem.constraints[0].violation()))

cpsr.cvxpy_solve(problem, verbose=False)
print('norm after refining with CPSR again: %.2e' % 
      np.linalg.norm(problem.constraints[0].violation()))
```

It has the following output. (Machine precision is around `1.11e-16`.)

```
norm of the constraint error, solving with SCS and then refining with CPSR: 1.48e-11
norm after refining with CPSR again: 5.21e-16
```
