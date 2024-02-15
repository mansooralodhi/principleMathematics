import ast
import copy
import astor

class InvTransform(ast.NodeTransformer):

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if isinstance(node.op, ast.Add):
            node.op = ast.Sub()
        elif isinstance(node.op, ast.Mult):
            node.op = ast.Div()
        """
        Official:
            Note that child nodes of nodes that have a custom visitor method, visit_[class-name],
            won’t be visited unless the visitor calls generic_visit() or visits them itself.
        Thus we call the parent class generic_visit() method. 
        The super() call is what propagates the tree traversal down the tree.
        In this case, it doesn't matter when is the supper called.
        """
        super().generic_visit(node)
        return node


expr = "f = 3*x + 4\nprint(f'output = {f}')"
tree = ast.parse(expr)  # create tree
actual_tree = copy.deepcopy(tree)

transformer = InvTransform()
# transform_tree = transformer.visit(tree)  # transform tree
transformer.visit(tree)  # transform tree


print('\n********** Print Transformed Tree ***********')
print(ast.dump(tree, annotate_fields=False, indent=1))

print("\n********** Unprase Transformed Tree *********")
print(ast.unparse(tree))

print("\n********** Back To Source Code *********")
print(astor.to_source(tree))

print("\n********** Compilation and Execution *******")
x = 1
exec(compile(actual_tree, '', 'exec'))
exec(compile(tree, '', 'exec'))