
import unittest

import scipy.linalg as slin
import numpy.testing as nt
from tools import simple_mul

import math

from padexp.exponential import Exponential
from padexp.polynomial import Polynomial

import numpy.linalg as lin
import numpy as np

def expm(M):
	"""
	Matrix exponential from :mod:`scipy`; adapt it to work on scalars.
	"""
	if np.isscalar(M):
		return math.exp(M)
	else:
		return slin.expm(M)


def phi_l(z, n=0):
	r"""
	Returns phi_l using the recursion formula:
		φ_0 = exp
		φ_{l+1}(z) = \frac{φ_l(z) - \frac{1}{l!}}{z}
	"""
	phi = expm(z)
	fac = Polynomial.exponents(z,0)[0] # identity
	if np.isscalar(z):
		iz = 1./z
	else:
		iz = lin.inv(z)
	for i in range(n):
		phi =  np.dot(phi - fac, iz)
		fac /= i+1
	return phi

phi_formulae = {
	0: lambda z: np.exp(z),
	1: lambda z: np.expm1(z)/z,
	2: lambda z: (np.expm1(z) - z)/z**2,
	3: lambda z: (np.expm1(z) - z - z**2/2)/z**3
}


def compare_phi_pade(computed, expected, phi):
	nt.assert_almost_equal(computed/expected, np.ones_like(expected), decimal=5)
	nt.assert_almost_equal(expected, phi)

def compare_to_id(res):
	nt.assert_array_almost_equal(res, np.identity(len(res)))

class TestExponential(unittest.TestCase):

	def test_phi_l(self):
		"""
		Check that :func:`phi_l` computes :math:`φ_l` correctly for scalars (by comparing to :data:`phi_formulae`).
		"""
		z = .1
		for n in range(4):
			expected = phi_formulae[n](z)
			computed = phi_l(z, n)
			nt.assert_almost_equal(computed, expected)

	def test_phi_0_mat(self):
		z = np.random.rand(2,2)
		expected = expm(z)
		computed = phi_l(z,0)
		nt.assert_almost_equal(computed, expected)

	def test_phi_1_mat(self):
		z = np.random.rand(2,2)
		expected = expm(z) - np.identity(2)
		expected = lin.solve(z, expected)
		computed = phi_l(z,1)
		nt.assert_almost_equal(computed, expected)

	def test_phi_pade(self,k=4,d=10):
		"""
		Test of the Padé approximation of :math:`φ_l` on matrices.
		"""
		phi = Exponential(k,d)
		Rs = phi.pade
		for z in  [.1*np.array([[1.,2.],[3.,1.]]),.1j*np.array([[1.j,2.],[3.,1.]]), np.array([[.01]]), np.array([[.1]])]:
			phis = phi(z)
			for n in range(1,k+1):
				R = Rs[n]
				N = R.numerator
				D = R.denominator
				expected = phi_l(z,n)
				Nz = simple_mul(N.coeffs, z)
				Dz = simple_mul(D.coeffs, z)
				computed = lin.solve(Dz,Nz)
				compare_phi_pade(computed, expected, phis[n]) # generate tests instead


	def test_identity(self,k=6, d=10):
		"""Test phi_k(0) = Id/k!"""
		z = np.zeros([2,2])
		phi = Exponential(k,d)
		phis = phi(z)
		for j,p in enumerate(phis):
			compare_to_id(p/phi.factorials[j]) # generate tests instead


	def test_phi_eval_pade_mat(self,k=8,d=6):
		z = .1*np.array([[1.,2.],[3.,1.]])
		phi = Exponential(k,d)
		computed = phi.eval_pade(z)[-1]
		expected = phi_l(z,k)
		nt.assert_almost_equal(computed, expected)

	def test_phi_scaled(self,n=5,d=10):
		z = 100.1
		phi = Exponential(n,d)
		expected = phi_l(z,n)
		computed = phi(z)[-1]
		nt.assert_approx_equal(computed, expected)

	def test_scaling(self,):
		nt.assert_equal(Exponential.scaling(1.1), 1)
		nt.assert_equal(Exponential.scaling(1.), 0)
		nt.assert_equal(Exponential.scaling(3.), 2)
		nt.assert_equal(Exponential.scaling(.1), 0)
		nt.assert_equal(Exponential.scaling(2.), 1)
		nt.assert_equal(Exponential.scaling(2.1), 2)

	def test_phi_scaled_mat(self,n=2,d=6):
		A =  np.array([[1.,2.],[3.,1.]])
	## 	z = np.random.rand(2,2)
		phi = Exponential(n,d)
		for z in [.01*A, .1*A, A, 2*A, 10*A]:
			expected = phi_l(z,n)
			computed = phi(z)[-1]
			nt.assert_almost_equal(computed/expected, np.ones_like(expected))

	def test_rotation(self, n=10):
		"""
		Test exp(w) with w n x n skew symmetric gives the right rotation matrix.
		"""
		J = np.zeros([n,n])
		J[0,1] = 1.
		J[1,0] = -1.
		e = Exponential()
		angle = np.random.rand()
		computed = e(angle*J)[0]
		expected = np.identity(n)
		expected[:2,:2] = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
		nt.assert_allclose(computed, expected)

	def test_translation(self, n=16):
		xi = np.zeros([n,n])
		t = np.random.rand(n-1)
		xi[:-1,-1] = t
		e = Exponential()
		computed = e(xi)[0]
		expected = xi.copy()
		expected += np.identity(n)
		nt.assert_allclose(computed, expected)
