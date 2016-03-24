# -*- coding: utf-8 -*-
"""

Test if contrib BKZ classes can be instantiated and run.
"""
from copy import copy

from fpylll import IntegerMatrix
from fpylll.contrib.simple_bkz import BKZReduction as SimpleBKZ
from fpylll.contrib.simple_dbkz import DBKZReduction as SimpleDualBKZ
from fpylll.contrib.bkz import BKZReduction as BKZ
from fpylll import BKZ as fplll_bkz
from fpylll.util import set_random_seed

dimensions = (30, 40)


def make_integer_matrix(m, n):
    A = IntegerMatrix(m, n)
    A.randomize(algorithm="ntrulike", bits=30, q=2147483647)
    return A


def test_bkz_init():
    for cls in (SimpleBKZ, SimpleDualBKZ, BKZ):
        for n in dimensions:
            set_random_seed(2**10 + n)
            A = make_integer_matrix(n, n)
            B = cls(copy(A))
            del B


def test_simple_bkz_call(block_size=10):
    for cls in (SimpleBKZ, SimpleDualBKZ):
        for n in dimensions:
            set_random_seed(n)
            A = make_integer_matrix(n, n)
            cls(A)(block_size=block_size)


def test_bkz_call(block_size=10):
    params = fplll_bkz.Param(block_size=block_size, flags=fplll_bkz.VERBOSE)
    for cls in (BKZ, ):
        for n in dimensions:
            set_random_seed(n)
            A = make_integer_matrix(n, n)
            B = copy(A)
            cls(B)(params=params)
