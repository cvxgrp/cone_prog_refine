# CPSR: Cone Program Solution Refinement

CPSR is a pure-Python library for the refinement of solutions of 
primal-dual cone programs. To install, execute in a terminal:

```
pip install cpsr
```

CPSR depends on `numpy` for vector arithmetic, 
`scipy` for sparse linear algebra,
and `numba` for just-in-time code compilation.

A detailed description of the algorithm used is provided
in the accompanying 
[paper](http://stanford.edu/~boyd/papers/cone_prog_refine.html),
which you can cite if you find the program useful.

A simple way to use CPSR is by solving a [CVXPY](https://www.cvxpy.org) problem
```
import cvxpy as cvx
import cpsr

problem = cvxpy.Problem( ... )

cpsr.cvxpy_solve(problem, presolve = True, verbose = True)
```
the objects of the CVXPY variables  will then contain
the result of the refinement algorithm.