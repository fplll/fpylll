# -*- coding: utf-8 -*-
"""

Test if Python BKZ classes can be instantiated and run.
"""
from copy import copy

from fpylll import IntegerMatrix
from fpylll.algorithms.simple_bkz import BKZReduction as SimpleBKZ
from fpylll.algorithms.simple_dbkz import DBKZReduction as SimpleDualBKZ
from fpylll.algorithms.bkz import BKZReduction as BKZ
from fpylll import BKZ as fplll_bkz
from fpylll.util import set_random_seed

dimensions = (31, 37)


def make_integer_matrix(n):
    A = IntegerMatrix.random(n, "ntrulike", bits=30)
    return A


def test_bkz_init():
    for cls in (SimpleBKZ, SimpleDualBKZ, BKZ):
        for n in dimensions:
            set_random_seed(2**10 + n)
            A = make_integer_matrix(n)
            B = cls(copy(A))
            del B


def test_simple_bkz_call(block_size=10):
    for cls in (SimpleBKZ, SimpleDualBKZ):
        for n in dimensions:
            set_random_seed(n)
            A = make_integer_matrix(n)
            cls(A)(block_size=block_size)


def test_bkz_call(block_size=10):
    params = fplll_bkz.Param(block_size=block_size, flags=fplll_bkz.VERBOSE|fplll_bkz.GH_BND)
    for cls in (BKZ, ):
        for n in dimensions:
            set_random_seed(n)
            A = make_integer_matrix(n)
            B = copy(A)
            cls(B)(params=params)
