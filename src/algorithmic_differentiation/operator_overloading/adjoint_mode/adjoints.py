

from src.algorithmic_differentiation.operator_overloading.adjoint_mode.node import Node
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.ops import *
"""
NB: we don't use the below adjoint methods independently and neither
    are they exposed to user, hence, rather then writing them as 
    simple functions and calling them using eval() we wrap them 
    inside class and call them using getattr() which is more safe 
    and logical. 
"""

# operand_b == rightOperand
# operand_a == leftOperand

class Adjoint(object):
    """
    method signature:
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

    @staticmethod
    def div(node: Node):
        return [
            node.adjoint / node.rightOperand.value,
            -1 * node.adjoint * node.leftOperand.value / node.rightOperand ** 2
        ]

    @staticmethod
    def pow(node: Node):
        return [
            node.adjoint * node.rightOperand * (node.leftOperand ** (node.rightOperand - 1)),
            node.adjoint * node * log(node.leftOperand)
        ]

    @staticmethod
    def transpose(node: Node):
        return [node.adjoint.T, None]