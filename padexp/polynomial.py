#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
import math

class Polynomial(object):
	"""
Polynomial class used in the Padé approximation.
Usage is as follows::

	p = Polynomial([4.,2.,1.,...]) # polynomial 4 + 2x + x**2 + ...
	Z = Polynomial.exponents(z, 3) # compute z**2 and z**3
	p(Z) # value of the polynomial p(z)

The evaluation at a matrix value is computed using the Paterson and Stockmeyer method (see [Golub]_ § 11.2.4).
The polynomial is split into chunks of size :data:`s`.

.. [Golub] Golub, G.H.  and van Loan, C.F., *Matrix Computations*, 3rd ed. :isbn:`9780801854149`
	"""
	def __init__(self, coeffs):
		self.coeffs = list(coeffs)

	@classmethod
	def exponents(self, z, s=1):
		"""
Compute the first s+1 exponents of z.

:Returns:
	[I,z,z**2,...,z**s]
		"""
		if np.isscalar(z):
			ident = 1
		else:
			ident = np.identity(len(z), dtype=z.dtype)
		Z = [ident]
		for i in range(s):
			Z.append(np.dot(Z[-1],z))
		return Z

	def __call__(self, Z):
		"""
Evaluate the polynomial on a matrix, using matrix multiplications (:func:`dot`).

:param list[s+1] Z:
		list of exponents of z up to s, so ``len(Z) == s+1``, where :data:`s` is the size of the chunks
		the various cases are the following:

		:s=1: Horner method
		:s ≥ d: naive polynomial evaluation.
		:s ≈ sqrt(d): optimal choice

		"""
		p = self.coeffs
		P = 0
		s = len(Z) - 1
		if s == 0: # ok only if the polynomial is constant
			if len(p) > 1:
				raise ValueError("Z must be provided in order to evaluate a non-constant polynomial.")
			return p[0]*Z[0]
		r = int(math.ceil(len(p)/s))
		# assert len(p) <= r*s # this should pass
		for k in reversed(range(r)):
			B = sum(b*Z[j] for j,b in enumerate(p[s*k:s*(k+1)]))
			P = np.dot(Z[s],P) + B
		return P
