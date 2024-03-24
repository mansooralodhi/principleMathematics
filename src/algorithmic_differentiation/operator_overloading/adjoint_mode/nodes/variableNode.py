import numpy as np
from typing import Union, Optional
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.absNode import AbsNode
from src.utilities.logger import log_new


class VariableNode(AbsNode):
    count = 0

    @log_new
    def __new__(cls, val: Union[float, int, np.ndarray], name: Optional[str] = None):
        if not isinstance(val, np.ndarray):
            val = np.asarray(val, dtype=float)
        if name is None:
            name = "var_%i" % cls.count
            cls.count += 1
        obj = super().__new__(cls, val.shape, val.dtype, buffer=val, strides=val.strides)
        obj.name = name
        return obj

    def __str__(self):
        return self.name


if __name__ == "__main__":
    from utils import print_node

    nodes = [VariableNode(45.0), VariableNode(34.5), VariableNode(-2.0)]
    for node in nodes:
        print(node)
        # print_node(node)
    # nodeA = VariableNode(3)
    # print(nodeA.count)
    # nodeB = VariableNode(5)
    # print(nodeB.count)
    # print("*" * 35)
    # VariableNode.count = 0
    # nodeA = VariableNode(3)
    # print(nodeA.count)
    # nodeB = VariableNode(5)
    # print(nodeB.count)
