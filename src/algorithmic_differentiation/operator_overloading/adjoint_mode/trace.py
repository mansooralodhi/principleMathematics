
from typing import Sequence
from collections import deque, defaultdict
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.adjoints import Node
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.adjoints import Adjoint

"""
NB: nodes in computational graph traced through
    breadth-first-search (BFS) algorithm which is performed 
    with the help of data structured called queue.
    queue uses the concept of first-in-first-out (FIFO) and
    fulfills the requirement of otherwise recursion. 
    with queue.append we are inserting at the end of queue
    and with queue.popleft we are removing from the start 
    of queue, thus each layer of graph is visited completely
    first before moving to next layer (or set of operations).  
"""


def trace_adjoints(endNode: Node) -> defaultdict:
    nodes = list()
    adjointOps = Adjoint()
    queue = deque()

    endNode.adjoint = 1
    queue.append(endNode)
    adjoints = defaultdict(None)

    while queue:

        currNode = queue.popleft()

        if currNode.opName is None:
            # variable or constant
            if 'const' in currNode.nodeName:
                currNode.adjoint = 0
            adjoints[currNode.nodeName] = currNode.adjoint
            # print("-" * 35)
            # print("Current node: ", currNode)
            # print("Node Adjoint: ", currNode.adjoint)
            # print("Node value: ", currNode.value)
            # print("-" * 35)
            continue

        # print("-" * 35)
        # print("Current node: ", currNode)
        # print("Node Adjoint: ", currNode.adjoint)
        # print("Node value: ", currNode.value)
        # print("Right operand: ", currNode.rightOperand)
        # print("Left operand: ", currNode.leftOperand)
        # print("-" * 35)

        leftAdjoint, rightAdjoint = getattr(adjointOps, currNode.opName)(currNode)

        currNode.leftOperand.adjoint += leftAdjoint
        currNode.rightOperand.adjoint += rightAdjoint

        nodes.append(currNode.nodeName)

        if currNode.rightOperand is not None:
            queue.append(currNode.rightOperand)
        if currNode.leftOperand is not None:
            queue.append(currNode.leftOperand)

    return adjoints


def trace_nodes(endNode: Node) -> Sequence[str]:
    # using breadth-first-traversal
    nodes = list()
    queue = deque()
    queue.append(endNode)
    while queue:
        currNode = queue.popleft()
        print("-" * 35)
        print("Current node: ", currNode)
        print("Node value: ", currNode.value)
        print("Right operand: ", currNode.rightOperand)
        print("Left operand: ", currNode.leftOperand)
        print("-" * 35)
        nodes.append(currNode.nodeName)
        if currNode.rightOperand is not None:
            queue.append(currNode.rightOperand)
        if currNode.leftOperand is not None:
            queue.append(currNode.leftOperand)
    return nodes


if __name__ == "__main__":
    import numpy as np
    x1 = Node(2.0, 'x1')
    x2 = Node(4.0, 'x2')
    x3 = Node(4.0, 'x3')
    node = (4*x1 - x2*x3) * (x1*x2 + x3)
    print(node.value)
    print(trace_adjoints(node))
