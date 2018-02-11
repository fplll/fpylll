# -*- coding: utf-8 -*-
"""

Test if Python BKZ classes can be instantiated and run.
"""
from copy import copy

from fpylll import IntegerMatrix, LLL
from fpylll.algorithms.simple_bkz import BKZReduction as SimpleBKZ
from fpylll.algorithms.simple_dbkz import DBKZReduction as SimpleDualBKZ
from fpylll.algorithms.bkz import BKZReduction as BKZ
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.tools.bkz_stats import BKZTreeTracer

from fpylll import BKZ as fplll_bkz
from fpylll import FPLLL

dimensions = (31, 37)


def make_integer_matrix(n):
    A = IntegerMatrix.random(n, "ntrulike", bits=30)
    return A


def test_bkz_init():
    for cls in (SimpleBKZ, SimpleDualBKZ, BKZ, BKZ2):
        for n in dimensions:
            FPLLL.set_random_seed(2**10 + n)
            A = make_integer_matrix(n)
            B = cls(copy(A))
            del B


def test_simple_bkz_call(block_size=10):
    for cls in (SimpleBKZ, SimpleDualBKZ):
        for n in dimensions:
            FPLLL.set_random_seed(n)
            A = make_integer_matrix(n)
            cls(A)(block_size=block_size)


def test_bkz_call(block_size=10):
    params = fplll_bkz.Param(block_size=block_size, flags=fplll_bkz.VERBOSE|fplll_bkz.GH_BND)
    for cls in (BKZ, BKZ2):
        for n in dimensions:
            FPLLL.set_random_seed(n)
            A = make_integer_matrix(n)
            B = copy(A)
            cls(B)(params=params)


def test_bkz_postprocessing():
    A = IntegerMatrix.random(20, "qary", bits=20, k=10, int_type="long")
    LLL.reduction(A)

    bkz = BKZ(A)
    bkz.M.update_gso()
    tracer = BKZTreeTracer(bkz)

    solution = (2, 2, 0, 3, 4, 5, 7)

    v = A.multiply_left(solution, 3)
    bkz.svp_postprocessing(3, len(solution), solution, tracer)
    w = tuple(A[3])
    assert v == w

    solution = (2, 1, 0, 3, 4, 5, 7)

    v = A.multiply_left(solution, 3)
    bkz.svp_postprocessing(3, len(solution), solution, tracer)
    w = tuple(A[3])
    assert v == w
