import numpy as np
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.operationNode import InternalOperationNode


class AbsNode(np.ndarray):
    """
    Input nodes: ConstantNode &/or VariableNode
    Output node: InternalOperationNode
    Child nodes: ConstantNode, VariableNodes
    """
    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)

    def __add__(self, other):
        val = np.add(self, other)
        return InternalOperationNode(val, opName='add', leftOperand=self, rightOperand=other)


if __name__ == '__main__':
    a = AbsNode((2, 3))
    b = AbsNode((2, 3))
    node = a + b
    # node = np.add(a, b)
    print(f"absNode: {node}")
    print(f"absNode Shape: {node.shape}")
    print(f"absNode Type: {type(node)}")
