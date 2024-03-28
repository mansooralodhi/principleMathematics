

import numpy as np
from collections import defaultdict


class Node(np.ndarray):
    """
    Remember:
            -   if 'opName' is None then node is a leaf (constant or variable) Node
    """

    constNodeCounter = 0
    opNodeCounter = defaultdict(int)

    def __new__(cls, val, nodeName: str = None, opName: str = None, leftOperand=None, rightOperand=None):

        if not isinstance(val, np.ndarray):
            val = np.asarray(val, dtype=float)

        obj = super().__new__(cls, val.shape, val.dtype, buffer=val, strides=val.strides)

        if nodeName is None:
            nodeId = cls.opNodeCounter[opName]
            nodeName = "%s_%d" % (opName, nodeId)
            cls.opNodeCounter[opName] += 1

        if nodeName == 'const':
            nodeName = "%s_%d" % (nodeName, cls.constNodeCounter)
            cls.constNodeCounter += 1

        obj.opName = opName
        obj.nodeName = nodeName
        obj.leftOperand = leftOperand
        obj.rightOperand = rightOperand
        obj.adjoint = 0.0

        return obj


    def __str__(self):
        return self.nodeName


    def __add__(self, other):
        return self._numpyOperation('__add__', 'add', self, other)

    def __sub__(self, other):
        return self._numpyOperation('__sub__', 'sub', self, other)

    def __mul__(self, other):
        return self._numpyOperation('__mul__', 'mul', self, other)

    def __truediv__(self, other):
        return self._numpyOperation('__truediv__', 'div', self, other)

    def __radd__(self, other):
        return self._numpyOperation('__radd__', 'add', self, other)

    def __rsub__(self, other):
        return self._numpyOperation('__rsub__', 'sub', self, other)

    def __rmul__(self, other):
        return self._numpyOperation('__rmul__', 'mul', self, other)

    def __rtruediv__(self, other):
        return self._numpyOperation('__rtruediv__', 'div', self, other)

    def __pow__(self, power, modulo=None):
        return self._numpyOperation('__pow__', 'pow', self, power)  # self ^ power

    def __rpow__(self, other):
        return self._numpyOperation('__rpow__', 'pow', self, other)  # other ^ self

    @property
    def T(self):
        val = np.transpose(self)
        return Node(val, opName='transpose', leftOperand=self)

    @property
    def value(self):
        return np.frombuffer(self.data, dtype=self.dtype)

    @staticmethod
    def _numpyOperation(funcName, opName, leftOperand, rightOperand):
        # todo: is it really necessary to create a constant value as a Node ?
        if not isinstance(rightOperand, Node):
            rightOperand = Node(val=rightOperand, nodeName='const')
        if not isinstance(leftOperand, Node):
            leftOperand = Node(val=leftOperand, nodeName='const')
        val = getattr(np.ndarray, funcName)(leftOperand, rightOperand)
        return Node(val, opName=opName, leftOperand=leftOperand, rightOperand=rightOperand)


if __name__ == '__main__':
    a = Node(3.0, 'x')
    b = Node(4.0, 'y')
    node = 7 * (a * (b + 2.0))
    print(f"Node: {node}")
    print(f"Node Shape: {node.shape}")
    print(f"Node Type: {type(node)}")
    print(f"Node Name: {node.nodeName}")
    print(f"Node Values: {node.tolist()}")
    print(isinstance(a, np.ndarray))
