import numpy as np
from function import Function
from plot import Plot


class NewtonMethod(Function, Plot):
    def __init__(self):
        Plot.__init__(self)
        Function.__init__(self)
        self.epsilon = np.finfo("float32").eps
        self.x_0 = np.float32(-20)

    def compute_next_x(self, curr_x_n):
        return curr_x_n - (self.compute_f_x(curr_x_n) / self.compute_f_x_derivative(curr_x_n))

    def compute_f_x_root(self):
        k = 1
        x_n = self.x_0
        while abs(self.compute_f_x(x_n)) > self.epsilon:
            print(f"k: {k}, x_n: {x_n}, f_x_n: {self.compute_f_x(x_n)}")
            x_n = self.compute_next_x(x_n)
            k += 1
        print(f"k: {k}, x_n: {x_n}, f_x_n: {self.compute_f_x(x_n)}")
        return x_n

    def compute_root_n_tangent(self):
        k = 1
        x_n = self.x_0
        f_x_n = self.compute_f_x(x_n)
        tangent_slope = tangent_y_intercept = 0
        while abs(f_x_n) > self.epsilon:
            print(f"k: {k}, x_n: {x_n}, f_x_n: {f_x_n}")
            x_n_next = self.compute_next_x(x_n)
            tangent_slope = f_x_n / (x_n - x_n_next)
            tangent_y_intercept = f_x_n - (tangent_slope * x_n)
            self.plot_iteration(k, x_n, f_x_n, self.compute_f_x,
                                self.compute_tangent, tangent_slope, tangent_y_intercept)
            f_x_n = self.compute_f_x(x_n_next)
            x_n = x_n_next
            k += 1
        self.plot_iteration(k, x_n, f_x_n, self.compute_f_x,
                            self.compute_tangent, tangent_slope, tangent_y_intercept)
        print(f"k: {k}, x_n: {x_n}, f_x_n: {f_x_n}")
        return x_n


if __name__ == '__main__':
    NewtonMethod().compute_root_n_tangent()
    NewtonMethod().compute_f_x_root()
