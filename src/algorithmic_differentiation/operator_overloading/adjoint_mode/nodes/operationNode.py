import numpy as np
from typing import Union, Optional

Any = Union[float, np.ndarray]


class InternalOperationNode(np.ndarray):
    opNodeCounter = {}

    def __new__(cls, val: np.ndarray, opName: str, leftOperand: Any, rightOperand: Any = None,
                nodeName: Optional[str] = None):
        if not isinstance(val, np.ndarray):
            val = np.asarray(val, dtype=float)
        obj = super().__new__(cls, val.shape, val.dtype, buffer=val, strides=val.strides)
        obj.opName = opName
        obj.leftOperand = leftOperand
        obj.rightOperand = rightOperand
        if nodeName is None:
            if opName not in cls.opNodeCounter:
                cls.opNodeCounter[opName] = 0
            nodeId = cls.opNodeCounter[opName]
            nodeName = "%s_%d" % (opName, nodeId)
            cls.opNodeCounter[opName] += 1
        obj.name = nodeName
        return obj


if __name__ == '__main__':
    nodes = []
    nodes.append(InternalOperationNode(np.asarray(21.0), 'mul', 7.0, 3.0))
    nodes.append(InternalOperationNode(np.asarray(81.0), 'mul', 9.0, 9.0))
    nodes.append(InternalOperationNode(np.asarray(81.0), 'mul', 9.0, 9.0))
    nodes.append(InternalOperationNode(np.asarray(3.0), 'div', 21.0, 7.0))
    for node in nodes:
        print(type(node))
        print(node.size)
        print(node.name)
        print(node.base)
        print("*" * 30)
