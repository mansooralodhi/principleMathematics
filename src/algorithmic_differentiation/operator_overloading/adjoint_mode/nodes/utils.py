


def print_node(node):
    print(f"Node Type:  {type(node)}")
    print(f"Node Shape:  {node.shape}")
    print(f"Node Size:  {node.size}")
    print(f"Node Dtype:  {node.dtype}")
    print(f"Node Value:  {node.tolist()}")
    print(f"Node Strides:  {node.strides}")
    print(f"Node Base:  {node.base}")
