# -*- coding: utf-8 -*-

from copy import copy

from fpylll import BKZ, IntegerMatrix, LLL
from fpylll.contrib.simple_bkz import BKZReduction as SimpleBKZ
from fpylll.util import set_random_seed

dimensions = (60, 70)


def make_integer_matrix(m, n):
    A = IntegerMatrix(m, n)
    A.randomize(algorithm="ntrulike", bits=30, q=2147483647)
    return A


def test_simple_bkz_init():
    for n in dimensions:
        set_random_seed(2**10 + n)
        A = make_integer_matrix(n, n)
        B = SimpleBKZ(copy(A))
        del B


def test_simple_bkz_reduction(block_size=10):
    for n in dimensions:
        set_random_seed(n)
        A = make_integer_matrix(n, n)
        LLL.reduction(A)
        B = copy(A)
        BKZ.reduction(B, BKZ.Param(block_size=block_size))

        C = copy(A)
        SimpleBKZ(C)(block_size=block_size)

        assert abs(C[0].norm() < A[0].norm())
        assert abs(C[0].norm() - B[0].norm()) < 0.1
