from src.algorithmic_differentiation.operator_overloading.ops import *
from src.algorithmic_differentiation.operator_overloading.verify_grad import *
from src.algorithmic_differentiation.operator_overloading.dual_number import DualNumber

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


# todo:
#       1. build matrix-vector product based on DualNumber
#           - matrix should be computed based on primal values
#           - vector is the binary vector corresponding to
#             initialization or derivative w.r.t.
#       2. find higher-order derivatives

f = lambda x, y: x*y

x = DualNumber(10, 0)  # with respect to variable x
y = DualNumber(20, 1)  # with respect to variable y
z = DualNumber(5, 1)  # with respect to variable y
print(f(x, y))

ans = froward_difference(f, [10, 20], [0, 1])
print(ans)
