#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

from padexp.polynomial import *

def simple_mul(p, x):
	"""
	Numerical value of the polynomial at x
		x may be a scalar or an array
	"""
	X = Polynomial.exponents(x,len(p)-1)
	return sum(pk*xk for pk,xk in zip(p,X))

