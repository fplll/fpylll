# -*- coding: utf-8 -*-

from fpylll import GSO, IntegerMatrix, LLL
from fpylll.fpylll import float_types
from copy import copy

dimensions = ((0, 0), (1, 1), (2, 2), (3, 3), (10, 10), (50, 50), (60, 60))


def make_integer_matrix(m, n):
    A = IntegerMatrix(m, n)
    A.randomize("uniform", bits=m+n)
    return A


def test_gso_init():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        for float_type in float_types:
            M = GSO.Mat(copy(A), float_type=float_type)
            del M

            U = IntegerMatrix(m, m)
            M = GSO.Mat(copy(A), U=U, float_type=float_type)
            del M

            UinvT = IntegerMatrix(m, m)
            M = GSO.Mat(copy(A), U=U, UinvT=UinvT, float_type=float_type)
            del M


def test_gso_d():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        for float_type in float_types:
            M = GSO.Mat(copy(A), float_type=float_type)
            assert M.d == m


def test_gso_int_gram_enabled():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        for float_type in float_types:
            M = GSO.Mat(copy(A), float_type=float_type)
            assert M.int_gram_enabled is False
            assert M.transform_enabled is False

            M = GSO.Mat(copy(A), float_type=float_type, flags=GSO.INT_GRAM)
            assert M.int_gram_enabled is True
            assert M.transform_enabled is False

            if m and n:
                U = IntegerMatrix(m, m)
                M = GSO.Mat(copy(A), U=U, float_type=float_type)
                assert M.transform_enabled is True
                assert M.inverse_transform_enabled is False

                UinvT = IntegerMatrix(m, m)
                M = GSO.Mat(copy(A), U=U, UinvT=UinvT, float_type=float_type)
                assert M.transform_enabled is True
                assert M.inverse_transform_enabled is True


def test_gso_update_gso():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
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
            abs(r00[0]/r00[i] - 1.0) < 0.0001
            abs(re00[0]/re00[i] - 1.0) < 0.0001
            abs(g00[0]/g00[i] - 1.0) < 0.0001
