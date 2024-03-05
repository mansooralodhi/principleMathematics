


from node import Node
import numpy as np
from typing import Union, Optional


# FixMe:
#       create a separate class for initiating variableNode.
#       operationNode(), allOperationNodes().
#       do not store instances in a list.
#       but since the nodes are mutually exclusive right now
#       (unlike linkedlist) therefore we may do it for time being.


Any = Union[float, np.ndarray]

class OperationNode(Node):

    # let's keep count of all variable nodes created
    count=0
    name='parent'
    opNodeCounter = {}

    def __str__(self):
        return f"Node = {self.name}"

    @staticmethod
    def create_using(val: np.ndarray, opName: str, leftOperand: Any, rightOperand: Any=None,
                     nodeName: Optional[str]=None) -> 'OperationNode':
        """
        Args:
            oprName:
            oprResult:
            leftOperand:
            rightOperand:
            nodeName:
        Returns:
        """
        obj = OperationNode(val.shape, val.dtype, buffer = val, strides= val.strides)
        obj.opName = opName
        obj.leftOperand = leftOperand
        obj.rightOperand = rightOperand
        if nodeName is None:
            if opName not in OperationNode.opNodeCounter:
                OperationNode.opNodeCounter[opName] = 0
            nodeId = OperationNode.opNodeCounter[opName]
            nodeName = "%s_%d" % (opName, nodeId)
            OperationNode.opNodeCounter[opName] += 1
        obj.name = nodeName
        return obj


if __name__ == '__main__':
    nodes = []
    opsNodes = OperationNode(shape=(2,))
    print(opsNodes)
    nodes.append(opsNodes.create_using(np.asarray(21.0), 'mul', 7.0, 3.0))
    nodes.append(opsNodes.create_using(np.asarray(81.0), 'mul', 9.0, 9.0))
    nodes.append(opsNodes.create_using(np.asarray(3.0), 'div', 21.0, 7.0))
    for node in nodes:
        print(node)
        print(type(node))
