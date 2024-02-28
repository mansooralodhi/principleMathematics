

"""
Reference: https://mostafa-samir.github.io/auto-diff-pt2/

-   in reverse-mode or adjoint-mode, we can make tangent 1 of
    more than one variable. in other words, we can compute
    derivative of a single output  w.r.t all input variables.
-   unlike forward-mode where we are computing all outputs
    against a single input, in reverse-mode we are computing
    a single output againt all the input variables.
-   in case we have more than one output in reverse mode against
    same variable than we can sum the derivatives, formerly
    known as multivariate chain rule.
-   we traverse the computational graph using breadth-first
    approach in reverse-mode.

Q.  why do we need to worry about the computational graph
    in case of reverse-mode when in forward-mode the python
    interpreter automatically interprets the results using
    DMAS rule ?
A.  Please refer to the article.
"""