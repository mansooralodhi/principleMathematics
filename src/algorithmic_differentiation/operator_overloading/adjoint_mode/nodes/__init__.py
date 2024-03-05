
from src.algorithmic_differentiation.operator_overloading.adjoint_mode.nodes.baseNode import Node

"""
"""
"""
Numpy is the base computational engine for the building
computational graph for calculating derivatives.

Graph vertices are the nodes (leaves as variables or
constants, and internal nodes as maths operations).

The edge are represented using nodes attributes (pointers).

The Node class is the Base class that extends three types
of nodes in our upcoming computational graph:
    1.  Constant Node
    2.  Variable Node
    3.  Operation Node

Subclassing Numpy:
    https://numpy.org/doc/stable/user/basics.subclassing.html

"""