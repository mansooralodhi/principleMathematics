

import numpy as np
from src.utilities.logger import log_new

class OperationNode(np.ndarray):

    opNodeCounter = {}

    @log_new
    def __new__(cls, val: np.ndarray, opName: str,
                leftOperand: np.ndarray, rightOperand: np.ndarray = None, nodeName: str = None):

        if not isinstance(val, np.ndarray):
            raise Exception("Something wrong is going on!")

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

    def __str__(self):
        return self.name


if __name__ == '__main__':
    nodes = [OperationNode(np.asarray(21.0), 'mul', 7.0, 3.0), OperationNode(np.asarray(81.0), 'mul', 9.0, 9.0),
             OperationNode(np.asarray(81.0), 'mul', 9.0, 9.0), OperationNode(np.asarray(3.0), 'div', 21.0, 7.0)]
    for node in nodes:
        print(type(node))
        print(node.size)
        print(node.name)
        print(node.base)
        print("*" * 30)
