

import numpy as np
from typing import Union, Optional
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.absNode import AbsNode


class VariableNode(AbsNode):

    count = 0

    def __new__(cls, val: Union[float, int, np.ndarray], name: Optional[str] = None):
        if not isinstance(val, np.ndarray):
            val = np.asarray(val, dtype=float)
        if name is None:
            name = "var_%i" % cls.count
            cls.count += 1
        obj = super().__new__(cls, val.shape, val.dtype, buffer=val, strides=val.strides)
        obj.name = name
        return obj


if __name__ == "__main__":
    # nodes = [VariableNode(45.0), VariableNode(34.5), VariableNode(-2.0)]
    # for node in nodes:
    #     print(type(node))
    #     print(node.size)
    #     print(node.name)
    #     print(node.base)
    #     print("*" * 30)
    nodeA = VariableNode(3)
    print(nodeA.count)
    nodeB = VariableNode(5)
    print(nodeB.count)
    print("*" * 35)
    VariableNode.count = 0
    nodeA = VariableNode(3)
    print(nodeA.count)
    nodeB = VariableNode(5)
    print(nodeB.count)
