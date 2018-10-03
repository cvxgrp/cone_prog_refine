from setuptools import setup

setup(
    name='cone_prog_refine',
    version='0.0.3',
    description='Solution refinement and perturbation analysis of conic programs.',
    author='E. Busseti, W. Moursi, S. Boyd',
    packages=['cone_prog_refine'],
    install_requires=['numpy', 'scipy', 'numba']
)
