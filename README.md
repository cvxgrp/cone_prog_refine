# Solution refinement of conic programs

We provide a reference implementation of the procedure developed in 
[our paper](https://stanford.edu/~boyd/papers/cone_prog_refine):

```
@article{cone_prog_refine,
  title={Solution Refinement at Regular Points of Conic Programs},
  author={E. Busseti and W. Moursi and S. Boyd},
  year={manuscript, 2018},
  howpublished = {\url{https://stanford.edu/~boyd/papers/cone_prog_refine}}
}
```

The code is in alpha (pre-release) state. We'll provide installation packages soon. 

To reproduce the experiments in the paper:
- make sure you have the necessary packages: Python 3, numpy, scipy, numba 
(we suggest using the (free) Python [Anaconda](https://www.anaconda.com/download) distribution, which includes these), 
- clone this repository
- launch a [jupyter](https://jupyter.org/) session and run the [experiments.ipynb](https://github.com/cvxgrp/cone_prog_refine/blob/master/examples/experiments.ipynb) notebook,
in the `examples` folder
