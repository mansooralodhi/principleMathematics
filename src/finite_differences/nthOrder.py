import numpy as np
from function import Function

"""
Important Take Away:
higher step size (machine precision/epsilon) does not always guarantee correct result.
especially if we go into higher order calculation (>3) then the step size needs to be 
high enough to even get a derivative. 
"""


class NthOrder(Function):

    def __init__(self, h=None):
        super().__init__()
        self.h = h if h is not None else np.finfo("float32").eps

    def central_difference(self, x, order, h=None):
        if h:
            self.h = h
        if order == 1:
            return self.compute_f_x(x)
        elif order == 2:
            return (self.compute_f_x(x + self.h) - self.compute_f_x(x - self.h)) / (2 * self.h)
        else:
            return (self.central_difference(x + self.h, order - 1) - self.central_difference(x - self.h, order - 1)) / (2 * self.h)


if __name__ == '__main__':
    x = 1
    differential_order = 5
    h = np.finfo("float32").eps # note: this higher precision will not give any result on order 4 or higher
    fintie_difference = NthOrder(h=1e-4)
    print(fintie_difference.central_difference(x, differential_order))
