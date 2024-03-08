

from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.ops import *
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.variableNode import VariableNode


x = VariableNode(0.5, 'x')
y = VariableNode(4, 'y')
z = VariableNode(-2.3, 'z')

f = sin( x ** (y + z)) - 3 * log((x**2) * (y ** 3))
ConstantNode(2)

print("f =  {}".format(f))