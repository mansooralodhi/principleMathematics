

from src.algorithmic_differentiation.operator_overloading.adjoint_mode.node import Node

"""
NB: we don't use the below adjoint methods independently and neither
    are they exposed to user, hence, rather then writing them as 
    simple functions and calling them using eval() we wrap them 
    inside class and call them using getattr() which is more safe 
    and logical. 
"""

class Adjoint(object):
    """
    method:
        Args:
            node during the reverse
            pass in computational graph
        Returns:
            adjoint of node's leftOperand
            adjoint of node's rightOperand
    """

    @staticmethod
    def add(node: Node):
        return [node.adjoint, node.adjoint]

    @staticmethod
    def sub(node: Node):
        return [node.adjoint, -1 * node.adjoint]

    @staticmethod
    def mul(node: Node):
        return [
            node.adjoint * node.rightOperand.value,
            node.adjoint * node.leftOperand.value,
        ]

