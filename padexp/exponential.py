# -*- coding: UTF-8 -*-
"""
:mod:`phi_pade` -- Exponentials with rational approximations
============================================================

Computation of φ functions using Padé approximations and Scaling and Squaring.

.. module :: phi_pade
	:synopsis: Computation of φ functions.
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>

Some formulae are taken from the `Expint documentation`_, of Håvard Berland, Bård Skaflestad and Will Wright.

.. _Expint project: http://www.math.ntnu.no/num/expint/
.. _Expint documentation: http://www.math.ntnu.no/preprint/numerics/2005/N4-2005.pdf


"""
from __future__ import division

import numpy as np
import math
import numpy.linalg as lin

from .rational import RationalFraction, Polynomial

def ninf(M):
	if np.isscalar(M):
		return abs(M)
	return lin.norm(M,np.inf)

class Exponential(object):
	r"""
Main class to compute the :math:`φ_l` functions.
The simplest way to define those functions is by the formula:

.. math::
	φ_{\ell}(x) = ∑_{k=0}^{∞} \frac{x^k}{(\ell+k)!}

Usage is as follows::

	phi = Exponential(k,d)
	result = phi(M)

where :data:`M` is a square array.
The variable ``result`` is a list of all the values of :math:`φ_{k}(M)` for :math:`0≤k≤l`.
	"""

	def __init__(self, k, d=6):
		self.k = k
		self.d = d
		self.pade = list(self.compute_Pade())

	def compute_Pade(self):
		r"""
Compute the Padé approximations of order :math:`d` of :math:`φ_l`, for :math:`0 ≤ l ≤ k`.

The goal is to produce an approximation of the form:

.. math::
	φ_{\ell} = \frac{N}{D}

where :math:`N` and :math:`D` are polynomials.
The formula for :math:`D` is first computed recursively using the following recursion relations:

.. math::
	D_0^0 = 1\\
	D_{j+1}^0 = \frac{-(d-j)}{(2d -j)(j+1)} D_{j}^0\\
	D_{j}^{\ell+1} = (2d-\ell)(2d+\ell+1-j) D_{j}^{\ell}

Then, considering the inverse factorial series:

.. math::
	C_j := \frac{1}{j!}

The numerator :math:`N` is now computed by:

.. math::
	N = D*C

		"""
		d = self.d
		k = self.k
		J = np.arange(d+1)
		j = J[:-1]
		a = -(d-j)/(2*d-j)/(j+1)
		D = np.ones([k+1,d+1])
		D[0,1:] = np.cumprod(a)
		l = np.arange(k).reshape(-1,1)
		al = (2*d-l)*(2*d+l+1-J)
		D[1:,:] = np.cumprod(al,0) * D[0,:]
		C = np.empty(d+k+2)
		C[0] = 1.
		C[1:] = 1./np.cumprod(np.arange(d+k+1)+1)
		self.C = C # save for future use; C[j] == 1/j!
		for m,Dr in enumerate(D):
			yield RationalFraction(np.convolve(Dr, C[m:m+d+1])[:d+1], Dr)

	@classmethod
	def scaling(self, z, threshold=0):
		norm = ninf(z)
		e = max(threshold, np.log2(norm))
		return int(math.ceil(e))

	def eval_pade(self, z, s=None):
		"""
		Evaluate :math:`φ_l(z)` using the Padé approximation.
		"""
		if s is None:
			s = int(math.floor(math.sqrt(self.d)))
		Rs = self.pade
		Z = Polynomial.exponents(z,s)
		phi = [R(Z) for R in Rs]
		return phi


	def __call__(self, z):
		"""
The final call to compute the values of :math:`φ_k(z)`.
It proceeds in three steps:

1. figure out a scaling exponent :math:`s` such that :math:`z/2^s` is reasonably little
2. compute :math:`φ_k(z/2^s)` using the Padé approximation
3. scale back by repeatedly squaring
		"""
		scaling = self.scaling(z)
		phis = self.eval_pade(z/2**scaling)
		for s in range(scaling):
			phis = self.square(phis)
		return phis

	def square_last(self, phis):
		"""
Formula for squaring phi_l from existing phi_k for k≤l, taken from the `Expint documentation`_.

The argument is an array containing [phi_0,...,phi_l].

.. _Expint documentation: http://www.math.ntnu.no/preprint/numerics/2005/N4-2005.pdf
		"""
		l = len(phis) - 1
		ifac = self.C
		odd = l % 2
		half = l//2
		next = half
		if odd:
			next += 1
		res = np.dot(phis[half], phis[next])
		res += sum(2*ifac[j]*phis[l-j] for j in xrange(half))
		if odd:
			res += ifac[half]*phis[half+1]
		res /= 2**l
		return res

	def square(self, phis):
		phis = [self.square_last(phis[:l+1]) for l in range(self.k+1)]
		return phis


