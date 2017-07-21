# -*- coding: utf-8 -*-

from fpylll import GSO, IntegerMatrix, BKZ, LLL
from fpylll.config import float_types
from copy import copy

dimensions = ((0, 0), (1, 1), (2, 2), (3, 3), (10, 10), (50, 50), (60, 60),)


def make_integer_matrix(m, n, int_type="mpz"):
    A = IntegerMatrix(m, n, int_type=int_type)
    A.randomize("uniform", bits=20)
    return A


def test_bkz_init():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        for float_type in float_types:
            M = GSO.Mat(copy(A), float_type=float_type)
            lll_obj = LLL.Reduction(M)
            param = BKZ.Param(block_size=3, strategies=BKZ.DEFAULT_STRATEGY)
            bkz = BKZ.Reduction(M, lll_obj, param)
            del bkz


def test_bkz_bkz():
    for m, n in dimensions:
        if m < 2 or n < 2:
            continue
        A = make_integer_matrix(m, n)
        b00 = []
        for float_type in float_types:
            B = copy(A)
            M = GSO.Mat(B, float_type=float_type)
            lll_obj = LLL.Reduction(M)
            param = BKZ.Param(block_size=min(m, 40), strategies=BKZ.DEFAULT_STRATEGY)
            bkz = BKZ.Reduction(M, lll_obj, param)
            bkz()
            b00.append(B[0, 0])
        for i in range(1, len(b00)):
            assert b00[0] == b00[i]
