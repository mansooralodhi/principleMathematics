from numbers import Number
from typing import Literal


"""                 General Differentiation Rules ! 
Product Rule: 
    d/dx (u.v)  =   u.v' +  u'.v
Quotient Rule:
    d/dx (u/v)  =   (u'.v - u.v') / v^2
Power Rule:
    d/dx (x^n)  =   n * x^(n-1)
Constant Rule:
    d/dx (c.f(x))  =   c . f'(x)
Sum Rule:
    d/dx (u + v)  = u' + v'  
Difference Rule:
    d/dx (u - v)  = u' - v'  
Chain Rule:
    d/dx (f(g(x))) = g'(x) . f'(g(x))

"""
######################  DataType: DualNumber #######################

class DualNumber:
    """
    All operations are based on differentiation rules.
    """
    def __init__(self, primal: Number, tangent: Literal[0, 1]):
        """
        Args:
            primal: the values of variable
            tangent: initialization/seeding of variable 1/0.
        Note:
            -   if variable is initialized as DualNumber with tangent 1,
                then derivative will be computed with respect to this variable.
        """
        self.primal = primal
        self.tangent = tangent

    def __repr__(self):
        return f"Dual(primal= {self.primal}, tangent= {self.tangent}))"

    def __mul__(self, other):
        return self._mul(self, other)

    def __rmul__(self, other):
        return self._mul(self, other)

    def __matmul__(self, other):
        pass

    def __add__(self, other):
        return self._add(self, other)

    def __radd__(self, other):
        return self._add(self, other)

    def __sub__(self, other):
        return self._sub(self, other)

    def __rsub__(self, other):
        return self._sub(self, other, True)

    @staticmethod
    def _mul(x, y):
        if isinstance(y, Number):
            # d/dx (c * f(x)) = f'(x)
            return DualNumber(primal=x.primal * y,
                              tangent=x.tangent * y)
        elif isinstance(y, DualNumber):
            # d/dx (u.v) = u.v' + u'.v
            return DualNumber(primal=x.primal * y.primal,
                              tangent=x.primal * y.tangent + x.tangent * y.primal)
        else: raise Exception(f"Operand of type {type(y)} is not supported.")

    @staticmethod
    def _add(x, y):
        if isinstance(y, Number):
            # d/dx (c + f(x)) = f'(x)
            return DualNumber(primal=x.primal + y,
                              tangent=x.tangent)
        elif isinstance(y, DualNumber):
            # d/dx (g(x) + f(x)) = g'(x) + f'(x)
            return DualNumber(primal=x.primal + y.primal,
                              tangent=x.tangent + y.tangent)
        else: raise Exception(f"Operand of type {type(y)} is not supported.")

    @staticmethod
    def _sub(x, y, reverse=False):
        if isinstance(y, Number):
            if reverse:
                # d/dx (c - f(x)) = f'(x)
                return DualNumber(primal=y - x.primal,
                                  tangent=-x.tangent)
            # d/dx (f(x) - c) = f'(x)
            return DualNumber(primal=x.primal - y,
                              tangent=x.tangent)
        elif isinstance(y, DualNumber):
            if reverse:
                # d/dx (g(x) - f(x)) = g'(x) - f'(x)
                return DualNumber(primal=y.primal - x.primal,
                                  tangent=y.tangent - x.tangent)
            # d/dx (f(x) - g(x)) = f'(x) - g'(x)
            return DualNumber(primal=x.primal - y.primal,
                              tangent=x.tangent - y.tangent)
        else: raise Exception(f"Operand of type {type(y)} is not supported.")

