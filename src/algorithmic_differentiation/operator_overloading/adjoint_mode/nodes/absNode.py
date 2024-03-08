

import numpy as np
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.constantNode import ConstantNode
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.operationNode import OperationNode
def _verifyInputDtype(other):
    if isinstance(other, AbsNode | ConstantNode | OperationNode):
        return other
    return ConstantNode(other)


class AbsNode(np.ndarray):
    """
    AbstractNode:
    Input nodes: ConstantNode | VariableNode | OperationNode
    Output node: OperationNode
    Child nodes: VariableNodes
    """

    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)


    def __add__(self, other):
        _verifyInputDtype(other)
        val = np.add(self, other)
        return OperationNode(val, opName='add', leftOperand=self, rightOperand=other)

    def __sub__(self, other):
        _verifyInputDtype(other)
        val = np.sub(self, other)
        return OperationNode(val, opName='sub', leftOperand=self, rightOperand=other)

    def __mul__(self, other):
        _verifyInputDtype(other)
        val = np.multiply(self, other)
        return OperationNode(val, opName='mul', leftOperand=self, rightOperand=other)

    def __truediv__(self, other):
        _verifyInputDtype(other)
        val = np.true_divide(self, other)
        return OperationNode(val, opName='div', leftOperand=self, rightOperand=other)

    def __radd__(self, other):
        _verifyInputDtype(other)
        val = np.add(other, self)
        return OperationNode(val, opName='add', leftOperand=other, rightOperand=self)

    def __rsub__(self, other):
        _verifyInputDtype(other)
        val = np.sub(other, self)
        return OperationNode(val, opName='sub', leftOperand=other, rightOperand=self)

    def __rmul__(self, other):
        _verifyInputDtype(other)
        val = np.multiply(other, self)
        return OperationNode(val, opName='mul', leftOperand=other, rightOperand=self)

    def __rtruediv__(self, other):
        _verifyInputDtype(other)
        val = np.true_divide(other, self)
        return OperationNode(val, opName='div', leftOperand=other, rightOperand=self)


if __name__ == '__main__':
    a = AbsNode((2, 3))
    b = AbsNode((2, 3))
    node = 5 + b - 3
    print(f"absNode: {node}")
    print(f"absNode Shape: {node.shape}")
    print(f"absNode Type: {type(node)}")

    print(isinstance(a, np.ndarray))