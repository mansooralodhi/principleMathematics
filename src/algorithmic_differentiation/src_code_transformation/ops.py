import ast

from src.algorithmic_differentiation.src_code_transformation.dual_node import DualNode


def add(op1: DualNode, op2: DualNode):
    return DualNode(primalNode=None)

def mul(op1: DualNode, op2: DualNode):
    primal = ast.BinOp(op1.primalNode, ast.Mult, op2.primalNode)
    if isinstance(op1.tangentNode, ast.Constant) and isinstance(op2.tangentNode, ast.Name):
        return DualNode(primalNode=primal, tangentNode=op2.tangentNode)
    elif isinstance(op1.tangentNode, ast.Name) and isinstance(op2.tangentNode, ast.Name):
        tang1 = ast.BinOp(op1.primalNode, ast.Mult, op2.tangentNode)
        tang2 = ast.BinOp(op1.tangentNode, ast.Mult, op2.primalNode)
        tangent = ast.BinOp(tang1, ast.Add, tang2)
        return DualNode(primalNode=primal, tangentNode=tangent)

    return
