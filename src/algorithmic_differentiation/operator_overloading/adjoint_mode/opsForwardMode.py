

import numpy as np
from src.utilities.logger import log_fun
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.graphNode import GraphNode

"""
Implemented Forward-Mode Operations 
(following GraphNode operations):
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

###############################  Preliminary Operations #######################
def _verifyInputDtype(other):
    if isinstance(other, GraphNode):
        return other
    return GraphNode(other)

@log_fun
def reset():
    GraphNode.opNodeCounter = 0
    GraphNode.constNodeCounter = 0

###############################  Trigonometric Operations #######################
@log_fun
def exp(array, name=None) -> GraphNode:
    # fixme: verify for computing adjoint
    _verifyInputDtype(array)
    val = np.exp(array)
    return GraphNode(val, opName='exp', leftOperand=array, nodeName=name)

@log_fun
def log(array, name=None):
    # fixme: verify for computing adjoint
    _verifyInputDtype(array)
    val = np.log(array)
    return GraphNode(val, opName='log', leftOperand=array, nodeName=name)

@log_fun
def sin(array, name=None) -> GraphNode:
    # fixme: verify for computing adjoint
    _verifyInputDtype(array)
    val = np.sin(array)
    return GraphNode(val, opName='sin', leftOperand=array, nodeName=name)

@log_fun
def cos(array, name=None):
    # fixme: verify for computing adjoint
    _verifyInputDtype(array)
    val = np.cos(array)
    return GraphNode(val, opName='cos', leftOperand=array, nodeName=name)

###############################  Reduction Operations #######################

@log_fun
def max(array, axis=None, keepdims=False, name=None) -> GraphNode:
    """
    i.e. max(shape=(3,2,4), axis=2, keepdims=False) -> (3,2)
         max(shape=(3,2,4), axis=2, keepdims=True) -> (3,2,1)
    NB: the node receives single input which is stored in leftOperand
    """
    _verifyInputDtype(array)
    val = np.max(array, axis=axis, keepdims=keepdims)
    opNode = GraphNode(val, opName='max', leftOperand=array, nodeName=name)
    # saving info for gradient
    opNode.axis = axis
    opNode.keep_dims = keepdims
    opNode.with_keepdims = np.max(array, axis=axis, keepdims=True)
    return opNode

@log_fun
def sum(array, axis=None, keepdims=False, name=None) -> GraphNode:
    """
    i.e. sum(shape=(2, 3, 4), axis=1, keepdims=False) -> (2,4)
         sum(shape=(2, 3, 4), axis=1, keepdims=True) -> (2,1,4)
    NB: the node receives single input which is stored in leftOperand
    """
    _verifyInputDtype(array)
    val = np.sum(array, axis=axis, keepdims=keepdims)
    opNode = GraphNode(val, opName='sum', leftOperand=array, nodeName=name)
    # saving info for gradient
    opNode.axis = axis
    opNode.keep_dims = keepdims
    opNode.with_keepdims = np.max(array, axis=axis, keepdims=True)
    return opNode

@log_fun
def mean(array, axis=None, keepdims=False, name=None) -> GraphNode:
    # todo: ...
    _verifyInputDtype(array)
    val = np.mean(array, axis=axis)
    # fixme: verify for computing adjoint
    opNode = GraphNode(val, opName='mean', leftOperand=array, nodeName=name)
    # saving info for gradient
    opNode.axis = axis
    opNode.keep_dims = keepdims
    opNode.with_keepdims = np.mean(array, axis=axis, keepdims=True)
    return opNode

@log_fun
def dot(array_a, array_b, name=None) -> GraphNode:
    # fixme: verify for computing adjoint
    _verifyInputDtype(array_a)
    _verifyInputDtype(array_b)
    val = np.dot(array_a, array_b)
    return GraphNode(val, opName='dot', leftOperand=array_a, rightOperand=array_b, nodeName=name)

@log_fun
def squeeze(array, axis=None, name=None):
    # fixme: verify for computing adjoint
    _verifyInputDtype(array)
    val = np.squeeze(array, axis=axis)
    return GraphNode(val, opName='squeeze', leftOperand=array, nodeName=name)

@log_fun
def softmax_cross_entropy(logits, labels, name=None):
    # fixme: verify for computing adjoint
    _verifyInputDtype(logits)
    _verifyInputDtype(labels)
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_op = np.exp(logits - logits_max)
    logits_softmax = exp_op / np.sum(exp_op, axis=1, keepdims=True)
    cross_entropy = -1 * np.mean(labels * np.log(logits_softmax + 1e-7))
    opNode = GraphNode(cross_entropy, opName='softmax_cross_entropy', leftOperand=logits, nodeName=name)
    # saving info for gradient
    opNode.softmax_val = logits_softmax
    opNode.labels = labels
    return opNode

###############################  Condition Operations #######################
@log_fun
def where(condition: np.ndarray, array_a, array_b, name=None) -> GraphNode:
    # fixme: verify for computing adjoint
    _verifyInputDtype(array_a)
    _verifyInputDtype(array_b)
    val = np.where(condition, array_a, array_b)
    opNode = GraphNode(val, opName='where', leftOperand=array_a, rightOperand=array_b, nodeName=name)
    # saving info for gradient
    opNode.condition = condition
    return opNode

###############################  Utilities Operations #######################

@log_fun
def reshape(array, new_shape: tuple, name=None):
    _verifyInputDtype(array)
    new_arr = np.reshape(array, new_shape)
    # fixme: verify for computing adjoint
    return GraphNode(new_arr, opName='reshape', leftOperand=array, nodeName=name)



if __name__ == "__main__":
    x = np.random.randint(0, 10, (3,2,4))
    x = GraphNode(x)
    print("x.shape : ", x.shape)
    y = sum(x, axis=1)
    print("y.shape : ", y.shape)