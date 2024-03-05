
import numpy as np

class BaseNode(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        return super().__new__(cls, shape, dtype, buffer, offset, strides, order)


if __name__=='__main__':
    node = BaseNode((2,3))
    print(f"BaseNode: {node}")
    print(f"BaseNode Shape: {node.shape}")
    print(f"BaseNode Type: {type(node)}")