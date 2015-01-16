#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
import numpy.linalg as lin

from .polynomial import Polynomial

class RationalFraction(object):
	"""
Super simple Rational Function class, used to compute N(Z)/D(Z) for polynomials N and D.
	"""
	def __init__(self, numerator, denominator):
		"""
Initialize the object with numerator and denominator coefficients.
		"""
		self.numerator = Polynomial(numerator)
		self.denominator = Polynomial(denominator)

	def __repr__(self):
		return "{0} / {1}".format(repr(self.numerator), repr(self.denominator))

	def __call__(self, Z):
		try:
			return lin.solve(self.denominator(Z), self.numerator(Z))
		except lin.LinAlgError:
			return self.numerator(Z)/self.denominator(Z)


