# Padé approximations of exponential (phi) functions

[![Build Status](https://img.shields.io/travis/olivierverdier/padexp/master.svg?style=flat-square)](https://travis-ci.org/olivierverdier/padexp)
[![Coverage Status](https://img.shields.io/coveralls/olivierverdier/padexp/master.svg?style=flat-square)](https://coveralls.io/r/olivierverdier/padexp?branch=master)
![Python version](https://img.shields.io/badge/python-2.7, 3.4-blue.svg?style=flat-square)

Exponential functions, in a general sense, are defined as

E_j(x) = ∑_k x^k/(k+j)!

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
This code is useful for exponential integrators, and is a port of the [expint Matlab package](http://www.math.ntnu.no/num/expint/matlab.php).
