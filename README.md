# `cpsr` (Cone Program Solution Refinement)

`cpsr` is a Python library 
for the iterative improvement, or refinement,
of a primal-dual solution,
or a certificate of unboundedness or infeasibility,
of a cone, or convex, program.
It operates by differentiating the conic optimality conditions,
and so it can also be used for *calculus* with conic programs.

### Refinement

Given an approximate solution (or certificate), 
meaning one for which the optimality 
conditions don't hold exactly, 
`cpsr` produces a new solution for which 
the norm of the violations of the primal and dual systems, 
and the duality gap, is smaller. 


**Mathematics.**
It does so by locally linearizing
the operator 𝒩 (z) ∈ 𝗥^(n),
the concatenation of the violations of the 
primal and dual systems of the problem, and the duality gap,
for any approximate primal-dual solution represented by z ∈ 𝗥^(n),
via an [embedding](https://www.jstor.org/stable/3690376) 
of the conic [optimality conditions](https://arxiv.org/pdf/1312.3039.pdf);
z can also represent a certificate, and in that case 𝒩 (z)
is the violation of its (primal or dual) system, concatenated with zero.
So, 𝒩 (z) = 0 if and only if z is an exact primal-dual solution
or certificate, meaning one for which the conic optimality conditions, or the certificate conditions,
are satisfied within machine precision. 
`cpsr` proceeds iteratively, using at each steps the current value of 𝒩 (z)
and the derivative matrix 𝗗𝒩 (z) to (approximately) solve
a linear system that locally approximates 
the conic optimality conditions. 

**Matrix-free.**
`cpsr` is a matrix-free solver, meaning that it does not store or
invert the derivative matrix 𝗗𝒩 (z). This allows it to scale
to very large problems. Essentially, if you are able to load the problem
data in memory, then `cpsr` can solve it, using O(n) additional memory, 
where n is the size of a primal-dual solution.

**Iterative solution.**
`cpsr` uses [LSQR](http://web.stanford.edu/group/SOL/software/lsqr/),
an iterative linear system solver, to approximately solve a system
that locally approximates the conic optimality conditions. 
The number of LSQR iterations is chosen by the user (by default, for small problems, 30),
as is the number of `cpsr` iterations (by default, for small problems, 2). 

**Problem classes.**
`cpsr` can currently solve cone programs whose cone constraints are products of 
the [zero cone](https://en.wikipedia.org/wiki/System_of_linear_equations),
[the non-negative orhant](https://en.wikipedia.org/wiki/Linear_programming),
and any number of [second-order cones](https://en.wikipedia.org/wiki/Second-order_cone_programming), 
[exponential cones](https://yalmip.github.io/tutorial/exponentialcone/), 
and [semidefinite cones](https://en.wikipedia.org/wiki/Semidefinite_programming).

**Paper.**
A much more detailed description of the algorithm used is provided
in the [accompanying paper](http://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf).
I show the experiments described in the paper in the
Jupyter notebook
[examples/experiments.ipynb](examples/experiments.ipynb).

**Dependencies.**
`cpsr` depends on [`numpy`](http://www.numpy.org) for vector arithmetics, 
[`scipy`](https://www.scipy.org) for sparse linear algebra,
and [`numba`](https://numba.pydata.org) for just-in-time code compilation.
It currently runs on a single thread, on CPU. 

**Future.**
I plan to rewrite the core library in C, 
support problems whose data is provided as an abstract linear operator,
and (possibly) provide a distributed implementation of the cone projections 
and the sparse matrix multiplications, either on multiple threads or on GPU.
I also plan to release interfaces to other scientific programming languages.


### Installation
To install, execute in a terminal:

```
pip install cpsr
```


### `cvxpy` interface

`cpsr` can be used in combination with [`cvxpy`](https://www.cvxpy.org),
via the `cpsr.cvxpy_solve` method. 
An example follows.

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
