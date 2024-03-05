


from node import Node
import numpy as np
from typing import Union, Optional


# FixMe:
#       create a separate class for initiating variableNode.
#       constantNode(), allConstantNodes().
#       do not store instances in a list.
#       but since the nodes are mutually exclusive right now
#       (unlike linkedlist) therefore we may do it for time being.



class ConstantNodes(Node):

    # let's keep count of all constant nodes created
    count=0
    name='parent'

    def __str__(self):
        return f"Node = {self.name}"

    @staticmethod
    def create_using(val: Union[float, np.ndarray], name: Optional[str]=None)-> 'ConstantNodes':
        """
        Args:
            val: number | list | ndarray
            name: node attribute or number
        Returns:
            obj
        """
        if isinstance(val, float):
            val = np.array(val, dtype=float)
        if name is None:
            name = "const_%i" % ConstantNodes.count
            ConstantNodes.count += 1
        obj = ConstantNodes(val.shape, val.dtype, buffer = val, strides=val.strides)
        obj.name = name
        return obj

if __name__ == '__main__':
    nodes = []
    constNodes = ConstantNodes(shape=(2,))
    print(constNodes)
    nodes.append(constNodes.create_using(45.0))
    nodes.append(constNodes.create_using(34.5))
    nodes.append(constNodes.create_using(-2.0))
    for node in nodes:
        print(node)
