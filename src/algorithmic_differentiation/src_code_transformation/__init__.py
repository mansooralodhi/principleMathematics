"""

Source:
    Main:
        Building Tree Interpreter:  https://deepsource.com/blog/python-asts-by-building-your-own-linter

    Supplementary:

        Parse Tree:                 https://dev.to/mblayman/deciphering-python-how-to-use-abstract-syntax-trees-ast-to-understand-code-gfm
        Transform a GraphNode:           https://tobiaskohn.ch/index.php/2018/07/30/transformations-in-python/
        Tree in conjunction with
        operator overloading        https://towardsdatascience.com/build-your-own-automatic-differentiation-program-6ecd585eec2a

    Built-in-Libraries:
        Tangent:                    https://github.com/google/tangent/tree/master


Notes on AST:

-   Ast module can be used to create, modify, and run ASTs from python code.
-   Everything in an ast is a node.
-   Nodes can be broadly classified as:
            - Literals
            - Variables
            - Statements
            - Expressions
-   In python, ast graph stores their exact location in the source code, this can
    be accessed through 'lineno' and 'col_offset' parameters.
-   At the top level, is a 'Module'. All Python files are compiled as "modules" when
    making the AST. Modules have a very specific meaning: anything that can be run
    by Python classifies as a module. So by definition, our Python file is a module.
-   Each module has a 'Body' which is nothing more than a list of statements.

"""