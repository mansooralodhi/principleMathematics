import numpy as np
from typing import Callable

# todo:
#       1. wrapp forward_difference with verify_gradient


def froward_difference(function: Callable, *args) -> dict[str, str]:
    """
    Args:
        function: Callable
        *args: tuple(arg1, arg2, ...)
    Returns: dict[primal, derivative]
    Note:
        Current implementation finds derivative w.r.t all variables.
    """
    h = np.finfo("float32").eps
    step = [arg + h for arg in args]
    primal = function(*args)
    derivative = (function(*step) - primal) / h
    # derivative = round(derivative, 5)
    return dict(primal=primal, derivative=derivative)





if __name__=="__main__":
    f = lambda x, y: x * y
    x = 10
    y = 20
    ans = froward_difference(f, x, y)
    print(ans)

