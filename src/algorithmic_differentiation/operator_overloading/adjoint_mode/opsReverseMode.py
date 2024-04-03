

from src.algorithmic_differentiation.operator_overloading.adjoint_mode.opsForwardMode import *
"""
NB: we don't use the below adjoint methods independently and neither
    are they exposed to user, hence, rather then writing them as 
    simple functions and calling them using eval() we wrap them 
    inside class and call them using getattr() which is more safe 
    and logical. 
"""
"""
Operations can be categorized as, reduction or arithmetic, when dealing with ndarrays.
Np basics:
    np.ones_like(x) restore the shape and dtype of input x while producing a matrix of ones. 
"""

# todo: work on the below operations:
"""
Implemented Reverse-Mode Operations:
    1. add
    2. sub
    3. mul
    4. div
    5. pow
    6. transpose
    7. exp
    8. log
    9. sin
   10. cos
   11. max
   12. sum
   13. mean
   14. dot
   15. squeeze
   16. softmax
   17. where 
   18. reshape 
"""

class Adjoint(object):
    """
    method signature:
        Args:
            node during the reverse-mode in computational graph
        Returns:
            adjoint of node's leftOperand
            adjoint of node's rightOperand
    Following the blog:
        operand_b == rightOperand
        operand_a == leftOperand

    """

    ######################################## Arithmetic Operations ################################
    @staticmethod
    def add(node: GraphNode):
        """
        u + v = adjoint * u, adjoint * v
        """
        return [node.adjoint, node.adjoint]

    @staticmethod
    def sub(node: GraphNode):
        """
        u - v = adjoint * u, - adjoint * v
        """
        return [node.adjoint, -1 * node.adjoint]

    @staticmethod
    def mul(node: GraphNode):
        """
        u . v = adjoint * u, adjoint * v
        """
        return [
            node.adjoint * node.rightOperand.value,
            node.adjoint * node.leftOperand.value,
        ]

    @staticmethod
    def div(node: GraphNode):
        """
        u/v = adjoint/v, -(adjoint * u) / (v^2)
        """
        return [
            node.adjoint / node.rightOperand.value,
            -1 * node.adjoint * node.leftOperand.value / node.rightOperand ** 2
        ]

    @staticmethod
    def pow(node: GraphNode):
        """
        x^n = adjoint * n * x^(n-1), ...
        n^x = ... , adjoint * n^x * log(n)
        """
        return [
            node.adjoint * node.rightOperand * (node.leftOperand ** (node.rightOperand - 1)),
            node.adjoint * node * log(node.leftOperand)
        ]

    @staticmethod
    def transpose(node: GraphNode):
        # todo: verify this rule for computing adjoint
        """
        x.T = x.T, None
        """
        return [node.adjoint.T, None]

    ######################################## Reduction Operations ################################

    @staticmethod
    def sum(node: GraphNode):
        """
        @param node: the output of sum operation during forward-mode
        @return: [ node: ones_like(shape=node.shape) , ]
        """
        return [node.adjoint * np.ones_like(node), None]

    @staticmethod
    def max(node: GraphNode):
        """
        @param node:
            node.leftOperand: input to max node during forward-mode
            node.with_keepdims: max of input (during forward-mode) with keepdims=True
        @return:
            partialMaxValuesDerivative
            assert partialMaxValuesDerivative.shape == node.leftOperand.shape
        example:
        --------
            node.leftOperand.shape = (3, 2, 5)
            node.with_keepdims.shape = (3, 2, 1) if axis = 2
            maxValIndices.shape = (3, 2, 5)
            totMaxValues.shape = (3, 2, 1) if axis = 2
            partialMaxValuesDeriv.shape = (3, 2, 5) return
        """
        """"
        theory:
        ------
            the maximum value may not be a single value in a matrix/vector
            but there may be more than one maximum value and each maximum
            value contribute equally to the final maximum value.
                i.e max([1, 2, 0, 2, 1, 2])
                    has max value of 2 but there are
                    more than one value equal to 2 and both contribute equally
                    to the final max value 2.
            hence, the adjoint or derivative is the combination of all
            partial derivatives of the maximum value, following the above
            example:
                adjoints = [0, 1/3, 0, 1/3, 0, 1/3]
            we can deduce that adjoint at max value is equal to
            1/(occurrences of max value) and 0 otherwise.
        """
        maxValIndices = np.where(node.leftOperand ==
                                 node.with_keepdims, 1, 0)
        totMaxValues = np.sum(maxValIndices, axis=node.axis, keepdims=True)
        partialMaxValuesDeriv = maxValIndices / totMaxValues
        return partialMaxValuesDeriv

    @staticmethod
    def mean(node: GraphNode):
        """
        theory:
        ------
        this node takes single input.
        adjoint of mean of a tensor is its .
        in-practise:
        -----------
        node -> mean of the input data to node during forward phase
        node.leftOperand -> original input data to node during forward phase
        """
        # fixme: the code is different from blog which was incorrect.
        maxValIndices = np.where(node.leftOperand == node.with_keepdims, 1, 0)
        totMaxValues = np.sum(maxValIndices, axis=node.axis, keepdims=True)
        partialMaxValuesDeriv = maxValIndices / totMaxValues
        return partialMaxValuesDeriv