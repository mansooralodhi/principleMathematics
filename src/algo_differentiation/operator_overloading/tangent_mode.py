from src.algo_differentiation.operator_overloading.ops import *
from src.algo_differentiation.operator_overloading.dual_number import DualNumber
from src.algo_differentiation.operator_overloading.utls import froward_difference

"""
Forward-Mode or Tangent-Mode
    1.  define function
    2.  initialize variables (primal values) 
    3.  seeding - 1/0        (tangent values)
    4.  extract function output
    5.  harvest (derivative w.r.t seed)
Note:   With DualNumbers we do initialization and seeding of variables at 
        sametime.
"""

f = lambda x, y: -1 * (sin(y) - cos(x))

x = DualNumber(10, 1)  # with respect to variable x
y = DualNumber(20, 1)  # with respect to variable y
print(f(x, y))

ans = froward_difference(f, 10, 20)
print(ans)
