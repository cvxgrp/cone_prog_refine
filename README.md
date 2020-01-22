# `cpr`: Cone program refinement

`cpr` is a Python library 
for the iterative refinement
of a primal-dual solution
(or a certificate of unboundedness or infeasibility)
of a cone program.

### Refinement

Given an approximate solution (or certificate), 
meaning one for which the optimality 
conditions don't hold exactly, 
`cpr` produces a new approximate solution for which 
the norm of the violations of the primal and dual systems, 
and the duality gap, is smaller. 


**Mathematics.**
It does so by differentiating
the operator 𝒩 (z) ∈ 𝗥^(n),
the concatenation of the violations of the 
primal and dual systems of the problem, and the duality gap,
for any approximate primal-dual solution
(or certificate) represented by z ∈ 𝗥^(n),
via an [embedding](https://www.jstor.org/stable/3690376) 
of the conic [optimality conditions](https://arxiv.org/pdf/1312.3039.pdf).
See the [accompanying paper](http://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf) for more details.
So, 𝒩 (z) = 0 if and only if z is an exact primal-dual solution
or certificate. 
`cpr` proceeds iteratively, using at each steps the value of 𝒩 (z)
and the derivative matrix 𝗗𝒩 (z) to approximately solve
the linear system that locally approximates the conic optimality conditions. 

**Matrix-free.**
`cpr` is matrix-free, meaning that it does not store or
invert the derivative matrix 𝗗𝒩 (z). 
In other words, `cpr` only uses O(n) memory space
in addition to the problem data, 
where n is the size of a primal-dual solution.

**Iterative solution.**
`cpr` uses [LSQR](http://web.stanford.edu/group/SOL/software/lsqr/),
an iterative linear system solver, to approximately solve a system
that locally approximates the conic optimality conditions. 
The number of LSQR iterations is chosen by the user (by default, for small problems, 30),
as is the number of `cpr` iterations (by default, for small problems, 2). 

**Problem classes.**
`cpr` can currently solve cone programs whose cone constraints are products of 
the [zero cone](https://en.wikipedia.org/wiki/System_of_linear_equations),
[the non-negative orhant](https://en.wikipedia.org/wiki/Linear_programming),
and any number of [second-order cones](https://en.wikipedia.org/wiki/Second-order_cone_programming), 
[exponential cones](https://yalmip.github.io/tutorial/exponentialcone/), 
and [semidefinite cones](https://en.wikipedia.org/wiki/Semidefinite_programming).

**Paper.**
A much more detailed description of the algorithm used is provided
in the [accompanying paper](http://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf).

**Dependencies.**
`cpr` depends on [`numpy`](http://www.numpy.org) for vector arithmetics, 
[`scipy`](https://www.scipy.org) for sparse linear algebra,
and [`numba`](https://numba.pydata.org) for just-in-time code compilation.
It currently runs on a single thread, on CPU. 

**Future plan.**
The core library will be rewritten in ANSI C,
and will 
support problems whose data is provided as 
an abstract linear operator.


### Installation
To install, execute in a terminal:

```
pip install cpr
```


### Minimal example (with `cvxpy`)

`cpr` can be used in combination with 
[`cvxpy`](https://www.cvxpy.org),
via the `cpr.cvxpy_solve` method. 

```python
import numpy as np
import cvxpy as cp
import cpr

n = 5
np.random.seed(1)
X = cp.Variable((n,n))

problem = cp.Problem(objective = cp.Minimize(cp.norm(X - np.random.randn(n, n))), 
                     constraints = [X @ np.random.randn(n) == np.random.randn(n)])

cpr.cvxpy_solve(problem, presolve=True, verbose=False)
print('norm of the constraint error, solving with SCS and then refining with CPSR: %.2e' % 
      np.linalg.norm(problem.constraints[0].violation()))

cpr.cvxpy_solve(problem, verbose=False)
print('norm after refining with CPSR again: %.2e' % 
      np.linalg.norm(problem.constraints[0].violation()))
```

It has the following output. (Machine precision is around `1.11e-16`.)

```
norm of the constraint error, solving with SCS and then refining with CPSR: 1.48e-11
norm after refining with CPSR again: 5.21e-16
```

### Citing

If you wish to cite `cpr`, please use the following BibTex:

```
@article{cpr2019,
    author       = {Busseti, E. and Moursi, W. and Boyd, S.},
    title        = {Solution refinement at regular points of conic problems},
    journal      = {Computational Optimization and Applications},
    year         = {2019},
    volume       = {74},
    number       = {3},
    pages        = {627--643},
}

@misc{cpr,
    author       = {Busseti, E. and Moursi, W. and Boyd, S.},
    title        = {{cpr}: cone program refinement, version 0.1},
    howpublished = {\url{https://github.com/cvxgrp/cone_prog_refine}},
    year         = 2019
}
```

