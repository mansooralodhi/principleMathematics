import ast


class Traverse(ast.NodeVisitor):
    """
    Depth-First Search - Pre-Order Traversal.
    Reference: https://www.guru99.com/tree-traversals-inorder-preorder-and-postorder.html
    """

    def visit(self, node):
        """
        Visit a node.
        Copied from ast.NodeVisitor.
        """
        method = 'visit_' + node.__class__.__name__
        print("Entering Node: ", method)
        # value = getattr(object, attribute_name, default_value)
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)


expr = 'f = 3*x + 4'  # "x=5"
tree = ast.parse(expr)  # create tree

print("\n*********** Print Tree****")
print(ast.dump(tree, annotate_fields=True, indent=1))

print("\n*********** Unprased ********")
print(ast.unparse(tree))
traverser = Traverse()

print("\n*********** Visit **********")
traverser.visit(tree)
