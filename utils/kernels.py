__author__ = "Felix Buettner"

import numpy as np


def linear_kernel(x1, x2, arg):
	"""Linear Kernel."""
	return np.dot(x1, x2)

def linear_spline_kernel(x1, x2, arg):
	"""Linear Spline Kernel."""
	x1 = x1[0]
	x2 = x2[0]
	m = min(x1, x2)
	k = 1 + x1*x2 + x1*x2*m - ((x1 + x2) / 2) * (m**2) + (m**3) / 3
	return k

def polynomial_kernel(x, y, p=2):
	"""Polynomial Kernel."""
	return np.power((np.dot(x, y) + 1), p)

def radial_kernel(x, y, gamma=0.1):
	"""Radial Kernel."""
	dif = np.subtract(x, y)
	dot = np.dot(dif, dif)
	return np.exp(np.multiply(-gamma, dot))


possible_kernel = {
	"lin": linear_kernel,
	"lsp": linear_spline_kernel,
	"pol": polynomial_kernel,
	"rbf": radial_kernel,
}