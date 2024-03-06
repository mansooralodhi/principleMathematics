import numpy as np
from lazy_import import lazy_module
# opNode = lazy_module("operationNode")
opNode = lazy_module("src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.operationNode")



class BaseNode(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    def __add__(self, other):
        # note: method not called if self is not BaseNode.
        val = np.add(self, other)
        return opNode.OperationNode(val, opName='add', leftOperand=self, rightOperand=other)


if __name__ == '__main__':
    a = BaseNode((2, 3))
    b = BaseNode((2, 3))
    node = a + b
    # node = np.add(a, b)
    print(f"BaseNode: {node}")
    print(f"BaseNode Shape: {node.shape}")
    print(f"BaseNode Type: {type(node)}")
