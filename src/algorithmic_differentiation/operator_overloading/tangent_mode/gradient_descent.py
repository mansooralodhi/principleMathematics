import numpy as np
from typing import List
from src.algorithmic_differentiation.operator_overloading.tangent_mode import main

# define a linear model:
y = lambda x: 1.4 * x - 7

# define a custom linear model
f = lambda x, p1, p2: x * p1 + p2

# define mean square error
mse = lambda x, p1, p2: np.square(y(x) - f(x, p1, p2))

# define learning rule
def update_params(x: float, params: List[float]):
    _x_grad, *params_grad = main.gradient_vector(mse)([x, *params])
    updated_params = np.asarray(params) - learning_rate * np.asarray(params_grad)
    return list(updated_params)

########## let's learn the right parameters for our custom linear model #########


# initialize model parameters
model_params = np.random.uniform(0, 1, (2,))
print(f"Initial model parameters: {model_params}")

# generate samples of input variable (x) and associated targets (y)
n = 5000
learning_rate = 0.01
Xs = np.random.uniform(0, 1, (n,))
Ys = np.asarray([y(i) for i in Xs])

for i in range(n):
    model_params = update_params(Xs[i], model_params)

print(f"Final model parameters: {model_params}")
print(f"Correct model parameters: [1.4, -7.0]")
