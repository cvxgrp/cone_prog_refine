from setuptools import setup

setup(
    name='cpsr',
    version='0.0.2',
    description='Cone Problem Solution Refinement.',
    author='Enzo Busseti',
    packages=['cpsr'],
    install_requires=['numpy>=1.15.1', 'scipy>=1.1.0', 'numba>=0.36.2']
)
