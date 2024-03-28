

"""
Reference: https://mostafa-samir.github.io/auto-diff-pt2/


-   unlike forward-mode where we are computing all outputs
    against a single input, in reverse-mode we are computing
    a single output againt all the input variables.
-   in case we have more than one output in reverse mode against
    same variable than we can sum the derivatives, formerly
    known as multivariate chain rule.
-   we traverse the computational graph using breadth-first
    approach in reverse-mode.

Q.  why do we need to worry about the computational graph
    in case of reverse-mode when in forward-mode the python
    interpreter automatically interprets the results using
    DMAS rule ?
A.  We need to know the computational graph or tree to know
    the sequence of operations and their operands to
    perform chain rule durin backpropagation of derivative.

Numpy is the base computational/operation engine
for the building directed acyclic graph for calculating
derivatives.

Graph vertices are the graph (leaves as variables or
constants, and internal graph as maths operations).

The edge are represented using graph attributes
(pointers). Nodes are connected through each node
attribute. Thus graph can be linked through:
leftOperand and rightOperand.


Whenever VariableNodes or (ConstantNode and VariableNodes)
interact, they call AbstractNode specialmethods that return the
OperationNode.

Subclassing Numpy:
    https://numpy.org/doc/stable/user/basics.subclassing.html
"""