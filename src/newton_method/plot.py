import numpy as np
import matplotlib.pyplot as plt


class Plot(object):
    def __init__(self):
        self.range = 20
        self.step_size = 1

    def set_range(self, x):
        start = x - self.range * self.step_size
        stop = x + (self.range + 1) * self.step_size
        return np.linspace(start, stop, int((stop - start) / self.step_size))

    def plot_iteration(self, k, x, f_x, x_func, tangent_func, m, c):
        # Calculate the start and stop values based on the base point and range size
        input = self.set_range(x)
        output = x_func(input)
        tangent = tangent_func(input, m , c)
        plt.plot(input, output, color='green', label="Function")
        plt.plot(input, tangent, color='blue', label="Tangent")
        plt.text(x+2, f_x-2, "k={:}\nx={:.6f}\nf={:.6f}".format(k, x, f_x))
        plt.axvline(x, color='black', linestyle='--')
        self.configure_plot()
        plt.show()

    @staticmethod
    def configure_plot():
        plt.legend(loc='upper right')  # Show legend with labels
        plt.title("Newton Method Demonstration")
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("f(x)")

