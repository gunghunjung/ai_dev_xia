import numpy as np
from scipy import integrate


def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def riemann_sum(f, a, b, n):
    """좌점 리만 합"""
    dx = (b - a) / n
    xs = np.linspace(a, b - dx, n)
    return np.sum(f(xs)) * dx


def exact_integral(f, a, b):
    result, _ = integrate.quad(f, a, b)
    return result
