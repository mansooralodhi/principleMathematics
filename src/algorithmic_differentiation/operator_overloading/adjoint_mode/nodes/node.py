
import numpy as np

class Node(np.ndarray):

    def __new__(cls, shape, *args, **kwargs):
        return super().__new__(cls, shape, *args, **kwargs)

    def __mul__(self, other):
        val = self * other



if __name__=='__main__':
    node = Node((2,3))
    print(f"Node: {node}")
    print(f"Node Shape: {node.shape}")
    print(f"Node Type: {type(node)}")