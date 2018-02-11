# -*- coding: utf-8 -*-

import sys
from fpylll import GSO, IntegerMatrix, LLL
from fpylll.config import float_types, int_types
from copy import copy

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
                assert abs(r00[0]/r00[i] - 1.0) < 0.0001
                assert abs(re00[0]/re00[i] - 1.0) < 0.0001
                assert abs(g00[0]/g00[i] - 1.0) < 0.0001


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
