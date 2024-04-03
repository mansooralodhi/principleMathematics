
from typing import Sequence
from collections import deque, defaultdict
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.opsReverseMode import GraphNode
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.opsReverseMode import Adjoint

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


def trace_adjoints(endNode: GraphNode) -> defaultdict:
    """
    this method only works for scalar-valued
    function. call this method iteratively for
    a vector-values function. and this is what
    actually happens for a vector-valued function.
    """

    queue = deque()
    adjointOps = Adjoint()

    endNode.adjoint = 1
    queue.append(endNode)
    adjoints = defaultdict(None)

    while queue:

        node = queue.popleft()

        if node.opName is None:
            # node is leaf node; constant or variable
            if not 'const' in node.nodeName:
                # node is variable node
                adjoints[node.nodeName] = node.adjoint
                # todo: node.adjoint = 0 -> is it necessary ?
            continue

        # apply chain rule
        leftAdjoint, rightAdjoint = getattr(adjointOps, node.opName)(node)

        node.leftOperand.adjoint += leftAdjoint
        node.rightOperand.adjoint += rightAdjoint

        if node.rightOperand is not None:
            queue.append(node.rightOperand)
        if node.leftOperand is not None:
            queue.append(node.leftOperand)

    return adjoints


def trace_nodes(endNode: GraphNode) -> Sequence[str]:
    # using breadth-first-traversal
    nodes = list()
    queue = deque()
    queue.append(endNode)
    while queue:
        node = queue.popleft()
        print("-" * 35)
        print("Current node: ", node)
        print("GraphNode value: ", node.value)
        print("Right operand: ", node.rightOperand)
        print("Left operand: ", node.leftOperand)
        print("-" * 35)
        nodes.append(node.nodeName)
        if node.rightOperand is not None:
            queue.append(node.rightOperand)
        if node.leftOperand is not None:
            queue.append(node.leftOperand)
    return nodes


if __name__ == "__main__":
    # import numpy as np
    # x = np.random.randint(0, 10, (2, 3))
    x1 = GraphNode(2.0, 'x1')
    x2 = GraphNode(4.0, 'x2')
    x3 = GraphNode(4.0, 'x3')
    node = (4*x1 - x2*x3) * (x1*x2 + x3) # forward-pass
    print("f: " , node.value)
    print("f': ", trace_adjoints(node)) # reverse-pass / adjoint-mode
