# -*- coding: utf-8 -*-
"""
Babai's Nearest Plane algorithm


..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>

"""

from fpylll import IntegerMatrix, LLL
from math import ceil


def babai(B, t):
    """
    Run Babai's Nearest Plane algorithm by running LLL.

    :param B: Input lattice basis.
    :param target: Target point (∈ ZZ^n)
    :returns coordinates of the solution vector:

    This implementation is more numerically stable compared to the one offered by `MatGSO.babai()`.
    On the other hand, this implementation will only accept tatgets with Integer coefficients.

    EXAMPLE ::

       >>> from fpylll import *
       >>> n = 10
       >>> B = IntegerMatrix(n, n + 1)
       >>> B.randomize("intrel", bits=100)
       >>> v_opt = B.multiply_left([1,0,1,0,1,1,0,0,1,1])
       >>> s = v_opt[0] # s = <a, x>, where a is vector of knapsack values.
       >>> t = [s] + (n * [0])
       >>> v = CVP.babai(B, t)
       >>> v[0] == t[0]
       True
       >>> v[1:]
       (1, 0, 1, 0, 1, 1, 0, 0, 1, 1)
       >>> _ = LLL.reduction(B)
       >>> v == CVP.closest_vector(B, t)
       True
    """
    A = IntegerMatrix(B.nrows + 1, B.ncols + 1)
    for i in range(B.nrows):
        for j in range(B.ncols):
            A[i, j] = B[i, j]

    # make sure the input is LLL reduced before reading the norm of the last vector
    LLL.reduction(A)
    # zero vector at the end
    A.swap_rows(0, B.nrows)

    for j in range(B.ncols):
        A[-1, j] = t[j]
    A[-1, -1] = ceil(A[-2].norm())

    LLL.reduction(A)  # now call LLL to run Babai

    v = [0] * len(t)
    if A[-1, -1] > 0:
        for i in range(len(t)):
            v[i] = t[i] - A[-1][i]
    else:
        for i in range(len(t)):
            v[i] = t[i] + A[-1][i]

    return tuple(v)
