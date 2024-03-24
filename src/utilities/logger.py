DEBUG = True

from typing import Callable


def log_new(f: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if False:
            print(f"Module: {f.__module__.split('.')[-1]}, Function: {f.__name__}, "
                  f"Args: ({args[0].__name__}, {args[1:]}), Kwargs: {kwargs}")
        return f(*args, **kwargs)

    return wrapper


def log_fun(f: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if DEBUG:
            result = f(*args, **kwargs)
            print(
                f"Module: {f.__module__.split('.')[-1]}, Function {f.__name__}, Input: {args, kwargs}, Output: {result}")
        return f(*args, **kwargs)

    return wrapper
