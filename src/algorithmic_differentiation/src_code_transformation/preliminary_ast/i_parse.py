
"""
[Optional] Tokenization:
    - in case of code (if-else, for-loop,) an addition step called tokenization is performed.
    - tokens are processed based on programming rules.
    - later, the tokens are parsed to form AST.

Parsing:
    - the process of translating a source program into an AST.
    - Nodes:
        -   BinOp:  binary operation,   fields: right | left | op
        -   Num:    number,             fields: n
        -   Name:   variable,           fields: id: name of the variable
                                        ctx (context) = Store(): store the value in variable             
                                                        Load(): use the stored value in variable    
                                                        Del(): delete the variable.

Direct AST Execution:
    - once AST is ready, it's possible to iterate over AST and execute the program.
    - Drawbacks:
        - AST might take large memory space.
        - AST time traversal might be longer.

Compilation:
    - due to the drawbacks of directly executing AST, the compiler converts the AST into bytecode.
    - compiler translates AST into Reverse Polish Notation (RPN) OR Polish Notation (PN)
        - source:   4*x + 3
        - RPN:      3 4 x * +  (since these 5 elements can be represented by a single byte, hence,
                                the resulting code is 'bytecode').
        - PN:       (+ 3 (* 4 x))

    - later, the bytecode is executed using 'stacked-based virtual machine'.
    - the 'python interpreter' executes this bytecode.
    - bytecode is used to call function in operating system and interact with CPU and memory.
"""

import ast
import astunparse   # pretty-print dump of ast

exp = "4*x + 3"
tree = ast.parse(exp)
print(astunparse.unparse(tree))
print(astunparse.dump(tree))

assign = "f = 4*x + 3"
tree = ast.parse(assign)
print(astunparse.unparse(tree))
print(astunparse.dump(tree))




