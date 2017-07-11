# -*- coding: utf-8 -*-

from fpylll import GSO, IntegerMatrix, LLL
from fpylll.config import float_types, int_types
from copy import copy

dimensions = ((0, 0), (1, 1), (2, 2), (3, 3), (10, 10), (50, 50), (60, 60),)


def make_integer_matrix(m, n, int_type="mpz"):
    A = IntegerMatrix(m, n, int_type=int_type)
    A.randomize("uniform", bits=m)
    return A


def test_lll_init():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        for float_type in float_types:
            M = GSO.Mat(copy(A), float_type=float_type)
            lll = LLL.Reduction(M)
            del lll


def test_lll_lll():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        for int_type in int_types:
            AA = IntegerMatrix.from_matrix(A, int_type=int_type)
            b00 = []
            for float_type in float_types:
                B = copy(AA)
                M = GSO.Mat(B, float_type=float_type)
                lll = LLL.Reduction(M)
                lll()
                if (m, n) == (0, 0):
                    continue
                b00.append(B[0, 0])
            for i in range(1, len(b00)):
                assert b00[0] == b00[i]
