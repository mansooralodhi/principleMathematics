from src.algorithmic_differentiation.operator_overloading.verify_grad import froward_difference
from src.algorithmic_differentiation.operator_overloading.tangent_mode.dualNumber import DualNumber
from typing import Callable, List, Literal
from functools import wraps

"""
Documentation
-------------
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

# todo:
#       1.  a limitation of current implementation is that it only accepts
#           f args as a flat list. In practise, the args can be any nested
#           data structure.
#           Future Goal:
#               1.  write a code to flatten the input arguments. e.g. flatten_tree
#               2.  write a code that produce a wrapper_function that operators
#                   on flattened arguments. similar to what is given below:
#                   from jax._src.api_util import flatten_fun_nokwargs

def directional_derivative(f: Callable) -> Callable[[List[Literal[0, 1]], List[float]], float]:
    @wraps(f)
    def wrapper(wrt_arg: List[Literal[0, 1]], args: List[float]) -> float:

        if len(wrt_arg) != len(args):
            raise Exception("len(wrt_arg) != len(args) !")

        dualNumbers = list()
        for i in range(len(args)):
            if wrt_arg[i] == 1:
                dualNumbers.append(DualNumber(args[i], 1))
            else:
                dualNumbers.append(DualNumber(args[i], 0))
        return f(*dualNumbers).tangent
    return wrapper


def gradient_vector(f: Callable) -> Callable[[List[float]], List[float]]:
    @wraps(f)
    def wrapper(args: List[float]):
        gradVector = list()
        for i in range(len(args)):
            wrt_args = [1 if i == j else 0 for j in range(len(args))]
            gradVector.append(directional_derivative(f)(wrt_args, args))
        return gradVector
    return wrapper


def total_derivative(f: Callable) -> Callable[[List[float]], float]:
    @wraps(f)
    def wrapper(args: List[float]):
        gradVector = gradient_vector(f)(args)
        return sum(gradVector)
    return wrapper


if __name__ == "__main__":
    func = lambda x, y: x * y
    x = DualNumber(10, 0)
    y = DualNumber(20, 1)  # with respect to variable y
    print(func(x, y))
    print(froward_difference(func, [10, 20], [0, 1]))
    print(directional_derivative(func)([0, 1], [10, 20]))
    print(directional_derivative(func)([1, 0], [10, 20]))
    print(gradient_vector(func)([10, 20]))
    print(directional_derivative(func)([1, 1], [10, 20]))
    print(total_derivative(func)([10, 20]))
