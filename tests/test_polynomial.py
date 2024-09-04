
import unittest
import numpy.testing as nt

from padexp.polynomial import *

from tools import simple_mul

class TestPolynomial(unittest.TestCase):
	def test_poly_exps(self):
		x = np.array([[1.,2.],[3.,1.]])
		x2 = np.dot(x,x)
		X = Polynomial.exponents(x,2)
		nt.assert_array_almost_equal(X[-1],x2)
		nt.assert_array_almost_equal(X[1],x)
		nt.assert_array_almost_equal(X[0], np.identity(2))

	def test_poly_exception(self):
		p = Polynomial([1.,2])
		x = np.array([[1.,2.],[3.,1.]])
		Z = Polynomial.exponents(x,0)
		with self.assertRaises(ValueError):
			p(Z)

	def test_poly_constant(self):
		p = Polynomial([1.])
		x = np.random.random_sample([2,2])
		Z = Polynomial.exponents(x,0)
		nt.assert_array_almost_equal(p(Z), np.identity(2))

	def test_simple_mul_mat(self):
		X = np.array([[1.,2.],[3.,1.]])
		expected = 9.*np.identity(2) + 3.*X + 2.*np.dot(X,X)
		computed = simple_mul([9.,3.,2.], X)
		nt.assert_almost_equal(computed, expected)


	def test_mat_pol(self,n=2):
		for d in range(1,20):
			p = Polynomial(np.random.rand(d+1))
			z = np.random.rand(n,n)
			expected = simple_mul(p.coeffs, z)
	## 		expected = p(Polynomial.exponents(z,1))
			for s in range(1, d+1):
				Z = Polynomial.exponents(z,s)
				computed = p(Z)
				nt.assert_almost_equal(computed, expected)

	def test_repr(self):
		p = Polynomial([1.,2.])
		expected = "Polynomial([1.0, 2.0])"
		self.assertEqual(repr(p), expected)

	def test_from_array(self):
		"""
		Coeffs are stored in a list.
		"""
		L = [1.,2.]
		p = Polynomial(np.array(L))
		self.assertEqual(p.coeffs, L)
