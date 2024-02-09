import numpy as np
from typing import Callable, Literal, List

# todo:
#       1. wrapp forward_difference with verify_gradient


def froward_difference(function: Callable, *args) -> dict[str, str]:
    """
    Note: This implementation finds derivative w.r.t all variables.
    Args:
        function: Callable
        *args: tuple(arg1, arg2, ...)
    Returns: dict[primal, derivative]
    """
    if len(args) == 0:
        raise Exception("Please provide initial variable values.")
    h = np.finfo("float32").eps
    step = [arg + h for arg in args]
    primal = function(*args)
    derivative = (function(*step) - primal) / h
    # derivative = round(derivative, 5)
    return dict(primal=primal, derivative=derivative)

def froward_difference(function: Callable, args: List, wrt_arg: List[Literal[0, 1]]) -> dict[str, str]:
    """
    Note: This implementation finds derivative w.r.t any number of variables.
    Args:
        function: Callable
        args: list: variables initialization
        wrt_arg: list: seeding - 0/1
            1: take derivative wrt this index variable in args
            0: don't take derivative wrt this index variable in args
    Constraint:
        len(args) == len(wrt_args)
    Returns: dict[primal, derivative]
    """
    if len(args) == 0:
        raise Exception("Please provide initial variable values.")
    if len(args) != len(wrt_arg):
        raise Exception("Number of args must be equal to number of wrt_args !")
    h = np.finfo("float32").eps
    step = list()
    for i in range(len(wrt_arg)):
        if wrt_arg[i] == 0:
            step.append(args[i])
        else:
            step.append(args[i]+h)
    primal = function(*args)
    derivative = (function(*step) - primal) / h
    # derivative = round(derivative, 5)
    return dict(primal=primal, derivative=derivative)


