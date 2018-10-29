"""
Enzo Busseti, Walaa Moursi, Stephen Boyd, 2018

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# these should be disabled when developing

ENABLE_JIT = False
ENABLE_NJIT = False

# ENABLE_JIT = True
# ENABLE_NJIT = True


identity_decorator = lambda x: x

jit = identity_decorator
njit = identity_decorator

if ENABLE_JIT:
    from numba import jit

if ENABLE_NJIT:
    from numba import njit
