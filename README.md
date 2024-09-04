# Padé approximations of exponential (phi) functions

[![Build Status](https://github.com/olivierverdier/padexp/actions/workflows/python_package.yml/badge.svg?branch=main)](https://github.com/olivierverdier/padexp/actions/workflows/python_package.yml?query=branch%3Amain)
![Python version](https://img.shields.io/badge/python-3.9,_3.10,_3.11,_3.12-blue.svg?style=flat-square)
[![codecov](https://codecov.io/github/olivierverdier/padexp/graph/badge.svg?token=Ea4XsTXw6A)](https://codecov.io/github/olivierverdier/padexp)


Exponential functions, in a general sense, are defined as

```math
E_j(x) = ∑_k x^k/(k+j)!
```

So for j=0, this is the regular exponential.

The main application is to apply that exponential function to a *matrix*.

Here is a minimal example:

```python
from padexp import Exponential
import numpy as np
e = Exponential(4) # to compute the functions E_j for 0 ≤ j ≤ 4
M = np.array([[1.,2.],[3.,4]])
e(M) # returns a list containing [E_0(M), E_1(M),...,E_4(M)]
```
This code is useful for exponential integrators, and is a port of the [expint Matlab package](https://dl.acm.org/doi/10.1145/1206040.1206044).
