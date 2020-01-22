from setuptools import setup

setup(
    name='cpr',
    version='0.1',
    description='Cone program refinement.',
    author='Enzo Busseti, Walaa Moursi, Stephen Boyd',
    packages=['cpr'],
    install_requires=['numpy>=1.15.1', 'scipy>=1.1.0', 'numba>=0.36.2']
)
