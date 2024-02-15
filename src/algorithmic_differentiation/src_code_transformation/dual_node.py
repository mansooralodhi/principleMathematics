
import ast

class DualNode(ast.AST):
    """
    DualNode of ast.Constant:
                primalNode = ast.Constant
                tangentNode = ast.Constant(0)
    DualNode of ast.Name:
                primalNode = ast.Name
                tangentNode = ast.Name(id=1) if ast.Name.id in wrt_arg else ast.Name(id=0)
    """

    def __init__(self, primalNode, tangentNode):
        super().__init__()
        self.primalNode = primalNode
        self.tangentNode = tangentNode

