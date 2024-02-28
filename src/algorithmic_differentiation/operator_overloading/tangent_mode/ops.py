import numpy as np
from typing import Union
from numbers import Number
from src.algorithmic_differentiation.operator_overloading.tangent_mode.dual_number import DualNumber


def sin(x: Union[DualNumber, Number]):
    if isinstance(x, Number):
        return np.sin(x)
    elif isinstance(x, DualNumber):
        return DualNumber(np.sin(x.primal), x.tangent * np.cos(x.primal))
    else: raise Exception(f"Operand of type {type(y)} is not supported.")


def cos(x: Union[DualNumber, Number]):
    if isinstance(x, Number):
        return np.cos(x)
    elif isinstance(x, DualNumber):
        return DualNumber(np.cos(x.primal), -1 * x.tangent * np.sin(x.primal))
    else: raise Exception(f"Operand of type {type(y)} is not supported.")
