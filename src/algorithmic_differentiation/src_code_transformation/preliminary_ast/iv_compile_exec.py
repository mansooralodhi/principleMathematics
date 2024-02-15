
import ast

expr = "f = 3*x + 4\nprint(f'output = {f}')"
tree = ast.parse(expr)  # create tree

print("\n********** Compilation and Execution *******")
x = 1
exec(compile(tree, '', 'exec'))