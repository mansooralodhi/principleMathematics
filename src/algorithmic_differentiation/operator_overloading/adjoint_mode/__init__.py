

"""
Reference: https://mostafa-samir.github.io/auto-diff-pt2/

#################### Basic Introduction ###########################
Q.  why do we need to worry about the computational graph
    in case of reverse-mode when in forward-mode the python
    interpreter automatically interprets the results using
    DMAS rule ?
A.  We need to know the computational graph or tree to know
    the sequence of operations and their operands to
    perform chain rule durin backpropagation of derivative.

-   unlike forward-mode where we are computing all outputs
    against a single input, in reverse-mode we are computing
    a single output against all the input variables in a
    single iteration.
-   in case we have more than one output in reverse mode against
    same variable than we can sum the derivatives, formerly
    known as multivariate chain rule.
-   we traverse the computational graph using breadth-first
    approach in reverse-mode.
-   numpy is the base computational/operation engine
    for the building directed acyclic graph for calculating
    derivatives.
-   graph vertices which are leaves act as variables or
    constants, and internal nodes act as maths operations.
-   the edge are represented using graph attributes
    (pointers: node.leftOperand). Nodes are connected through
    each node attribute. Thus graph can be linked through:
    leftOperand and rightOperand.

Subclassing Numpy:
    https://numpy.org/doc/stable/user/basics.subclassing.html

#################### Program Pipeline ###########################

"""