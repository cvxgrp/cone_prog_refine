from setuptools import setup

setup(
    name='cpsr',
    version='0.0.1',
    description='Cone Problem Solution Refinement.',
    author='E. Busseti',
    packages=['cpsr'],
    install_requires=['numpy>=1.15.1', 'scipy>=1.1.0', 'numba>=0.36.2']
)
