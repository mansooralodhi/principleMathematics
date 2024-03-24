

from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes import ops
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.variableNode import VariableNode


"""
Numpy is the base computational engine for the building
computational graph for calculating derivatives.

Graph vertices are the nodes (leaves as variables or
constants, and internal nodes as maths operations).

The edge are represented using nodes attributes (pointers).

The AbsNode class inherits np.ndarray and extends into 
variable node in our upcoming computational graph:

The Constant node is not extended from AbsNode but values
that not part of variable are made constantNode
during runtime.

A third type of node, which is not for user but only
internal use, inherits np.ndarray and represent an
operational node, called 'OperationNode'.
 
Thus, user only gets to interact with variableNode
that extends from AbsNode and the other type of
nodes are kept hidden from user.

Whenever VariableNodes or (ConstantNode and VariableNodes) 
interact, they call AbsNode specialmethods that return the 
InternalOperationNode. 

Subclassing Numpy:
    https://numpy.org/doc/stable/user/basics.subclassing.html

"""