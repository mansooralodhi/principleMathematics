from src.algo_differentiation.operator_overloading.ops import *
from src.algo_differentiation.operator_overloading.dual_number import DualNumber
from src.algo_differentiation.operator_overloading.verify_grad import *

"""
Forward-Mode or Tangent-Mode Algorithm
    1.  define function
    2.  initialize variables (primal values) 
    3.  seeding - 1/0        (tangent values)
    4.  extract function output
    5.  harvest (derivative w.r.t seed)
DualNumbers:   
    -   if variable is initialized as DualNumber with tangent 1,
        then derivative will be computed with respect to this variable.
    -   in forward-mode or tangent-mode, we can make tangent 1 of only
        one variable. in other words, we can compute derivative of
        all output functions w.r.t only single variable/input.

"""

f = lambda x, y, z: 4*x + 6*y - 3 * z

x = DualNumber(10, 1)  # with respect to variable x
y = DualNumber(20, 1)  # with respect to variable y
z = DualNumber(5, 1)  # with respect to variable y
print(f(x, y, z))

ans = froward_difference(f, [10, 20, 5], [1, 1, 1])
print(ans)
