# -*- coding: utf-8 -*-

import sys
import pytest

from fpylll import GSO, IntegerMatrix, LLL
from fpylll.config import float_types, int_types
from copy import copy

import tools


if sys.maxsize >= 2**62:
    dimensions = ((0, 0), (2, 2), (3, 3), (10, 10), (30, 30), (50, 50), (60, 60))
else:
    dimensions = ((0, 0), (2, 2), (3, 3), (10, 10), (30, 30))


def make_integer_matrix(m, n, int_type="mpz"):
    A = IntegerMatrix(m, n, int_type=int_type)
    A.randomize("qary", k=m//2, bits=m)
    return A


def test_gso_init():
    for int_type in int_types:
        for m, n in dimensions:
            A = make_integer_matrix(m, n, int_type=int_type)
            for float_type in float_types:
                M = GSO.Mat(copy(A), float_type=float_type)
                del M

                U = IntegerMatrix(m, m, int_type=int_type)
                M = GSO.Mat(copy(A), U=U, float_type=float_type)
                del M

                UinvT = IntegerMatrix(m, m, int_type=int_type)
                M = GSO.Mat(copy(A), U=U, UinvT=UinvT, float_type=float_type)
                del M


def test_gso_d():
    for int_type in int_types:
        for m, n in dimensions:
            A = make_integer_matrix(m, n, int_type=int_type)
            for float_type in float_types:
                M = GSO.Mat(copy(A), float_type=float_type)
                assert M.d == m


def test_gso_int_gram_enabled():
    for int_type in int_types:
        for m, n in dimensions:
            A = make_integer_matrix(m, n, int_type=int_type)
            for float_type in float_types:
                M = GSO.Mat(copy(A), float_type=float_type)
                assert M.int_gram_enabled is False
                assert M.transform_enabled is False

                M = GSO.Mat(copy(A), float_type=float_type, flags=GSO.INT_GRAM)
                assert M.int_gram_enabled is True
                assert M.transform_enabled is False

                if m and n:
                    U = IntegerMatrix(m, m, int_type=int_type)
                    M = GSO.Mat(copy(A), U=U, float_type=float_type)
                    assert M.transform_enabled is True
                    assert M.inverse_transform_enabled is False

                    UinvT = IntegerMatrix(m, m, int_type=int_type)
                    M = GSO.Mat(copy(A), U=U, UinvT=UinvT, float_type=float_type)
                    assert M.transform_enabled is True
                    assert M.inverse_transform_enabled is True


def test_gso_update_gso():
    EPSILON = 0.0001

    for int_type in int_types:
        for m, n in dimensions:
            A = make_integer_matrix(m, n, int_type=int_type)
            LLL.reduction(A)

            r00 = []
            re00 = []
            g00 = []
            for float_type in float_types:
                M = GSO.Mat(copy(A), float_type=float_type)
                M.update_gso()
                if (m, n) == (0, 0):
                    continue
                r00.append(M.get_r(0, 0))
                re00.append(M.get_r_exp(0, 0)[0])
                g00.append(M.get_gram(0, 0))

            for i in range(1, len(r00)):
                assert r00[0] ==  pytest.approx(r00[i], rel=EPSILON)
                assert re00[0] ==  pytest.approx(re00[i], rel=EPSILON)
                assert g00[0] ==  pytest.approx(g00[i], rel=EPSILON)


def test_gso_io():
    for int_type in int_types:
        for m, n in dimensions:
            if m <= 2 or n <= 2:
                continue

            A = make_integer_matrix(m, n, int_type=int_type)
            v = list(A[0])
            LLL.reduction(A)

            for float_type in float_types:
                M = GSO.Mat(copy(A), float_type=float_type)
                M.update_gso()
                w = M.babai(v)
                v_ = IntegerMatrix.from_iterable(1, m, w) * A
                v_ = list(v_[0])
                assert v == v_


def test_gso_coherence_gram_matrix():
    """
        Test if the GSO is coherent if it is given a matrix A or its associated
        Gram matrix A*A^T
    """

    EPSILON = 0.0001

    for m, n in dimensions:
        for int_type in int_types:
            # long is not tested for high dimensions because of integer overflow
            if m > 20 and int_type == "long":
                continue

            A = make_integer_matrix(m, n, int_type=int_type).transpose()
            G = tools.compute_gram(A)

            for float_type in float_types:
                M_A = GSO.Mat(copy(A), float_type=float_type, gram=False, flags=GSO.INT_GRAM)
                M_A.update_gso()

                M_G = GSO.Mat(copy(G), float_type=float_type, gram=True, flags=GSO.INT_GRAM)
                M_G.update_gso()

                # Check that the gram matrix coincide
                for i in range(m):
                    for j in range(i):
                        assert M_A.get_int_gram(i, j) == G[i, j]

                # Check if computations coincide
                for i in range(m):
                    M_A.get_r(i, i) == pytest.approx(M_G.get_r(i, j), rel=EPSILON)

                    for j in range(i):
                        assert (
                            M_A.get_r(i, j) == pytest.approx(M_G.get_r(i, j), rel=EPSILON)
                        )
                        assert (
                            M_A.get_mu(i, j) == pytest.approx(M_G.get_mu(i, j), rel=EPSILON)
                        )
