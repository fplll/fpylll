# -*- coding: utf-8 -*-

from fpylll import GSO, IntegerMatrix, LLL
from fpylll.config import float_types, int_types
from copy import copy

import sys
import tools

if sys.maxsize > 2 ** 32:
    dimensions = ((0, 0), (1, 1), (2, 2), (3, 3), (10, 10), (50, 50), (60, 60))
else:
    # https://github.com/fplll/fpylll/issues/112
    dimensions = ((0, 0), (1, 1), (2, 2), (3, 3), (10, 10), (20, 20), (30, 30))


def make_integer_matrix(m, n, int_type="mpz"):
    A = IntegerMatrix(m, n, int_type=int_type)
    A.randomize("qary", bits=20, k=min(m, n) // 2)
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


def test_lll_gram_lll_coherence():
    """
        Test if LLL is coherent if it is given a matrix A or its associated
        Gram matrix A*A^T

        We should have Gram(LLL_basis(A)) = LLL_Gram(Gram(A)).
    """

    for m, n in dimensions:
        for int_type in int_types:
            A = make_integer_matrix(m, n)
            G = tools.compute_gram(A)

            for float_type in float_types:
                M_A = GSO.Mat(A, float_type=float_type, gram=False)
                lll_A = LLL.Reduction(M_A)

                M_G = GSO.Mat(G, float_type=float_type, gram=True)
                lll_G = LLL.Reduction(M_G)

                # A modified in place
                lll_A()
                # G modified in place
                lll_G()

                G_updated = tools.compute_gram(A)

                if (m, n) == (0, 0):
                    continue
                for i in range(m):
                    for j in range(i + 1):
                        assert G_updated[i, j] == G[i, j]
