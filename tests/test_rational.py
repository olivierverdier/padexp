#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest
import numpy.testing as nt

import scipy.linalg as slin

from padexp.polynomial import *
from padexp.rational import *

class TestRational(unittest.TestCase):
	def test_rational_fraction(self):
		N = Polynomial([1.,2.])
		D = Polynomial([3.,4])
		R = RationalFraction(N.coeffs, D.coeffs)
		X = np.random.random_sample([2,2])
		Z = Polynomial.exponents(X)
		nt.assert_array_almost_equal(R(Z), slin.solve(D(Z),N(Z)))
		x = 3.
		z = Polynomial.exponents(x)
		nt.assert_array_almost_equal(R(z), N(z)/D(z))


