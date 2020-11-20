# -*- coding: utf-8 -*-

from fpylll import GSO, IntegerMatrix, LLL, SVP, Enumeration
import pytest

dimensions = ((3, 3), (10, 10), (20, 20),)


def make_integer_matrix(m, n):
    A = IntegerMatrix(m, n)
    A.randomize("uniform", bits=10)
    return A


def test_svp():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        A = LLL.reduction(A)
        M = GSO.Mat(A)
        M.update_gso()
        v0 = SVP.shortest_vector(A)

        E = Enumeration(M)
        _, v1 = E.enumerate(0, M.d, M.get_r(0, 0), 0)[0]
        v1 = A.multiply_left(v1)

        assert v0 == v1


def test_svp_too_large():
    from fpylll.config import max_enum_dim
    m = max_enum_dim + 1
    n = max_enum_dim + 1
    A = make_integer_matrix(m, n)
    A = LLL.reduction(A)
    M = GSO.Mat(A)
    M.update_gso()
    with pytest.raises(NotImplementedError):
        SVP.shortest_vector(A)
