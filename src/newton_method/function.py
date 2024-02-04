import numpy as np


class Function(object):
    def __init__(self):
        pass

    @staticmethod
    def compute_tangent(m, x, c):
        return (m*x) + c

    @staticmethod
    def compute_f_x(x_n):
        # # # 2x^4 + x^3 + 3x^2 + 1
        # return 2 * (x_n**4) + x_n**3 + 1
        # x^2 - 2
        return x_n ** np.float32(2) - np.float32(2)

    @staticmethod
    def compute_f_x_derivative(x_n):
        # 8x^3 + 3x^2
        # return 8 * (x_n ** 3)  + 3 * (x_n**2)
        # 2x
        return 2 * x_n

    @staticmethod
    def compute_f_x_second_derivative(x_n):
        return 2