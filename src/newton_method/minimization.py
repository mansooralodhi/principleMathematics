import numpy as np
from function import Function


class Minimization(Function):
    def __init__(self):
        Function.__init__(self)
        self.epsilon = np.finfo("float32").eps
        self.x_0 = np.float32(-20)

    def compute_next_x(self, curr_x_n):
        # Note: the reason we can rewrite the newton method, originally in terms of f, f`,
        # into terms of (f`, f``) is due to Taylor Series !
        return curr_x_n - (self.compute_f_x_derivative(curr_x_n) / self.compute_f_x_second_derivative(curr_x_n))

    def compute_minimum(self):
        k = 1
        x_n = self.x_0
        while abs(self.compute_f_x_derivative(x_n)) > self.epsilon:
            print(
                f"k: {k}, x_n: {x_n}, f_x_n: {self.compute_f_x(x_n)}, "
                f"f_x_n_derivative: {self.compute_f_x_derivative(x_n)}, "
                f"f_x_second_derivative: {self.compute_f_x_second_derivative(x_n)}")
            x_n = self.compute_next_x(x_n)
            k += 1
        print(
            f"k: {k}, x_n: {x_n}, f_x_n: {self.compute_f_x(x_n)}, "
            f"f_x_n_derivative: {self.compute_f_x_derivative(x_n)}, "
            f"f_x_second_derivative: {self.compute_f_x_second_derivative(x_n)}")
        print("Function is Convex !") if self.compute_f_x_second_derivative(x_n) > np.float32(0) else print(
            "Function is Concave !")

        return x_n


if __name__ == '__main__':
    Minimization().compute_minimum()
