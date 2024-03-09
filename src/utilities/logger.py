


from typing import Callable

def log_new(f):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {f.__name__}, Args: ({args[0].__name__}, {args[1:]}), Kwargs: {kwargs}")
        return f(*args, **kwargs)
    return wrapper

def log_fun(f: Callable):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        print(f"Function {f.__name__}, Input: {args, kwargs}, Output: {result}")
        return f(*args, **kwargs)
    return wrapper


