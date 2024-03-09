

from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes import *


x = VariableNode(0.5, 'x')
y = VariableNode(4, 'y')
z = VariableNode(-2.3, 'z')

f = ops.sin( x ** (y + z)) - 3 * ops.log((x**2) * (y ** 3))

print("f =  {}".format(f))