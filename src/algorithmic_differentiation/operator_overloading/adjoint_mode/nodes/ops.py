

import numpy as np
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.constantNode import ConstantNode
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.variableNode import VariableNode
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.operationNode import OperationNode
from src.utilities.logger import log_fun

def _verifyInputDtype(other):
    if isinstance(other, VariableNode | ConstantNode):
        return other
    return ConstantNode(other)


@log_fun
def reset():
    ConstantNode.count = 0
    VariableNode.count = 0
    OperationNode.opNodeCounter = {}

@log_fun
def sum(array, axis=None, keepdims=False, name=None) -> OperationNode:
    _verifyInputDtype(array)
    val = np.sum(array, axis=axis, keepdims=keepdims)
    return OperationNode(val, opName='sum', leftOperand=array, nodeName=name)

@log_fun
def mean(array, axis=None, name=None) -> OperationNode:
    _verifyInputDtype(array)
    val = np.mean(array, axis=axis)
    return OperationNode(val, opName='mean', leftOperand=array, nodeName=name)

@log_fun
def exp(array, name=None) -> OperationNode:
    _verifyInputDtype(array)
    val = np.exp(array)
    return OperationNode(val, opName='exp', leftOperand=array, nodeName=name)

@log_fun
def log(array, name=None):
    _verifyInputDtype(array)
    val = np.log(array)
    return OperationNode(val, opName='log', leftOperand=array, nodeName=name)

@log_fun
def dot(array_a, array_b, name=None) -> OperationNode:
    _verifyInputDtype(array_a)
    _verifyInputDtype(array_b)
    val = np.dot(array_a, array_b)
    return OperationNode(val, opName='dot', leftOperand=array_a, rightOperand=array_b, nodeName=name)


@log_fun
def sin(array, name=None) -> OperationNode:
    _verifyInputDtype(array)
    val = np.sin(array)
    return OperationNode(val, opName='sin', leftOperand=array, nodeName=name)


@log_fun
def cos(array, name=None):
    _verifyInputDtype(array)
    val = np.cos(array)
    return OperationNode(val, opName='cos', leftOperand=array, nodeName=name)

@log_fun
def reshape(array, new_shape: tuple, name=None):
    _verifyInputDtype(array)
    new_arr = np.reshape(array, new_shape)
    return OperationNode(new_arr, opName='reshape', leftOperand=array, name=name)


@log_fun
def squeeze(array, axis=None, name=None):
    _verifyInputDtype(array)
    val = np.squeeze(array, axis=axis)
    return OperationNode(val, opName='squeeze', leftOperand=array, name=name)

@log_fun
def softmax_cross_entropy(logits, labels, name=None):
    _verifyInputDtype(logits)
    _verifyInputDtype(labels)
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_op = np.exp(logits - logits_max)
    logits_softmax = exp_op / np.sum(exp_op, axis=1, keepdims=True)
    cross_entropy = -1 * np.mean(labels * np.log(logits_softmax + 1e-7))
    opNode = OperationNode(cross_entropy, opName='softmax_cross_entropy', leftOperand=logits, name=name)
    # todo: understand (note: this is a scalar-valued function)
    #  saving info for gradient
    opNode.softmax_val = logits_softmax
    opNode.labels = labels
    return opNode

@log_fun
def max(array, axis=None, keepdims=False, name=None) -> OperationNode:
    _verifyInputDtype(array)
    val = np.max(array, axis=axis, keepdims=keepdims)
    opNode = OperationNode(val, opName='max', leftOperand=array, nodeName=name)
    # todo: understand (note: this is a scalar-valued function)
    #  saving info for gradient
    opNode.axis = axis
    opNode.keep_dims = keepdims
    opNode.with_keepdims = np.max(array, axis=axis, keepdims=True)
    return opNode

@log_fun
def where(condition: np.ndarray, array_a, array_b, name=None) -> OperationNode:
    _verifyInputDtype(array_a)
    _verifyInputDtype(array_b)
    val = np.where(condition, array_a, array_b)
    opNode = OperationNode(val, opName='where', leftOperand=array_a, rightOperand=array_b, nodeName=name)
    # todo: understand (note: this is a scalar-valued function)
    #  saving info for gradient
    opNode.condition = condition
    return opNode