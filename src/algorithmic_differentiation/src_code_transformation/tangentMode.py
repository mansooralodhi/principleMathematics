import ast
from typing import Union, Tuple
from src.algorithmic_differentiation.src_code_transformation.dualNode import *



class TangentModeDerivative(ast.NodeTransformer):
    """
    Description:
        The derivative is computed essentially using chain rule.
        Each node is transformed is into DualNode which possess
        both the actual(primary) node and its derivative.
        At each BinOp, the chain rule is applied on DualNode.
        This is achieved using pre-order traversal to make
        sure each operand of BinOp gets a DualNode to compute
        derivative.
    Limitation:
        1.  considers only the expression and not the assignment.
            expressions that have nested BinOp only.
            the BinOp is expected to possess only ast.Name or ast.Constant.
        2.  once you transform a single ast node into DualNode then you
            have to transform all parent graph into DualNode as well.
    """

    def __init__(self, *wrt_var: Union[str, Tuple[str]]):
        self.wrt_var = wrt_var

    def visit_Name(self, node: ast.Name) -> DualNode:
        """
        Transform every variable (leaf node in ast) into DualNode
        Args:
            node:       ast.Name
        Returns:
            DualNode:   primalNode: node
                        tangentNode: ast.Name(id=1) if node.id in wrt_arg else ast.Name(id=0)
        """
        if node.id in self.wrt_var:
            return DualNode(primalNode=node, tangentNode=ast.Name(id=1, ctx=node.ctx))
        return DualNode(primalNode=node, tangentNode=ast.Name(id=0, ctx=node.ctx))

    def visit_Constant(self, node: ast.Constant) -> DualNode:
        """
        Transform every constant (leaf node in ast) into DualNode
        Args:
            node:       ast.Constant
        Returns:
            DualNode:   primalNode: node
                        tangentNode: ast.Constant(0)
        """
        return DualNode(primalNode=node, tangentNode=ast.Constant(value=0))

    def visit_BinOp(self, node: ast.BinOp) -> DualNode:
        """
        Perform chain rule over tangent node and simple
        binOp on the primal node.
        Args:
            node:
        Returns:
            DualNode:
                    primalNode: node.op(node.left.primal, node.right.primal)
                    tangentNode: chain-rule-op(node.left.tangent, node.right.tangent)
        """
        super().generic_visit(node)
        if not (isinstance(node.left, DualNode) and isinstance(node.right, DualNode)):
            raise Exception("Alert: tree traversal is not pre-order !!!")
        primal = ast.BinOp(left=node.left.primalNode, op=node.op, right=node.right.primalNode)
        if isinstance(node.op, ast.Add):
            tangent = ast.BinOp(left=node.left.tangentNode, op=node.op, right=node.right.tangentNode)
            return DualNode(primal, tangent)
        elif isinstance(node.op, ast.Mult):
            tangent = ast.BinOp(left=node.left.tangentNode, op=ast.Add(), right=node.right.tangentNode)
            return DualNode(primal, tangent)



if __name__ == "__main__":
    expr = "f = 3 * x\nprint(f'output = {f}')"
    tree = ast.parse(expr)

    differ = TangentModeDerivative('x')
    transformed_tree = differ.visit(tree)
    # transform_srcCode = astor.to_source(transformed_tree).strip()
    # print("Actual Source Code: ", expr)
    # print("Transformed Source Code: ", transform_srcCode)

    # print("\n********** Compilation and Execution *******")
    x = 4
    exec(compile(tree, '', 'exec'))
    # exec(compile(transformed_tree, '', 'exec'))
