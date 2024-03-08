

import numpy as np


class ConstantNode(np.ndarray):

    count = 0

    def __new__(cls, val, name: str = None):

        if not isinstance(val, np.ndarray):
            val = np.asarray(val, dtype=float)
        if name is None:
            name = "const_%i" % cls.count
            cls.count += 1

        obj = super().__new__(cls, val.shape, val.dtype, buffer=val, strides=val.strides)

        obj.name = name
        return obj


if __name__ == "__main__":
    from utils import print_node
    output = ConstantNode(45.0)+ ConstantNode(45.0)
    print_node(output)
    print("*" * 30)
    nodes = [ConstantNode(45.0), ConstantNode(34.5), ConstantNode(-2.0)]
    for node in nodes:
        print_node(node)
        print("*" * 30)

