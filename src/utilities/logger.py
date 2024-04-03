from typing import Callable

DEBUG_Obj = False
DEBUG_Ops = False

"""
Note:   there might be issues while using the below decorators. For e.g. the GraphNode counter may increment
        twice for the same operation because the operation will be called twice due to its decorator.
        this will happen even if the above DEBUG variables are made False, in order to get correct logs
        remove the decorators from functions. 
"""
def log_new(f: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if DEBUG_Obj:
            print(f"Module: {f.__module__.split('.')[-1]}, Function: {f.__name__}, "
                  f"Args: ({args[0].__name__}, {args[1:]}), Kwargs: {kwargs}")
        return f(*args, **kwargs)

    return wrapper


def log_fun(f: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if DEBUG_Ops:
            result = f(*args, **kwargs)
            print(
                f"Module: {f.__module__.split('.')[-1]}, Function {f.__name__}, Input: {args, kwargs}, Output: {result}")
        return f(*args, **kwargs)

    return wrapper
