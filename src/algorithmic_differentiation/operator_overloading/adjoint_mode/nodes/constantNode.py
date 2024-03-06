

import numpy as np
from typing import Union, Optional
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.absNode import AbsNode


class ConstantNode(AbsNode):

    count = 0

    def __new__(cls, val: Union[float, int, np.ndarray], name: Optional[str] = None):
        if isinstance(val, np.ndarray):
            pass
        elif isinstance(val, float | int):
            val = np.asarray(val, dtype=float)
        else:
            raise Exception("Invalid dtype of argument 'val'.")
        if name is None:
            name = "const_%i" % cls.count
            cls.count += 1
        obj = super().__new__(cls, val.shape, val.dtype, buffer=val, strides=val.strides)
        obj.name = name
        return obj


if __name__ == "__main__":
    nodes = []
    nodes.append(ConstantNode(45.0))
    nodes.append(ConstantNode(34.5))
    nodes.append(ConstantNode(-2.0))
    for node in nodes:
        print(type(node))
        print(node.size)
        print(node.name)
        print(node.base)
        print("*" * 30)
