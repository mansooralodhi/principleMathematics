from sympy import symbols, diff, sin, exp
import numpy as np


x = symbols('x')
expression = exp(sin(x**2))

# Compute the derivative with respect to x
derivative = diff(expression, x)
print(derivative)
