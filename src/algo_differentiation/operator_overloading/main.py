
from math import pi
import dmath as dmath
from mathematics.algo_differentiation.operator_overloading.forwardAD import derivative


# fx = lambda x: dmath.sin(x) ** dmath.sin(x)
fx = lambda x: x**2 * x **3

print("{:.10f}".format(derivative(fx, 0, [1])))  # prints 0.3616192241