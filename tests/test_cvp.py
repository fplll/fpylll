# -*- coding: utf-8 -*-

from fpylll import GSO, IntegerMatrix, LLL, CVP, Enumeration
import pytest

dimensions = ((3, 3), (10, 10), (20, 20), (40, 40),)


def make_integer_matrix(m, n):
    A = IntegerMatrix(m, n)
    A.randomize("uniform", bits=10)
    return A


def test_cvp_cvp():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        A = LLL.reduction(A)
        M = GSO.Mat(A)
        M.update_gso()
        t = list(make_integer_matrix(n, n)[0])
        v0 = CVP.closest_vector(A, t)

        E = Enumeration(M)
        _, v1 = E.enumerate(0, A.nrows, 2, 40, M.from_canonical(t))[0]
        v1 = IntegerMatrix.from_iterable(1, A.nrows, map(lambda x: int(round(x)), v1))
        v1 = tuple((v1*A)[0])

        assert v0 == v1


def test_cvp_too_large():
    from fpylll.config import max_enum_dim
    m = max_enum_dim + 1
    n = max_enum_dim + 1
    A = make_integer_matrix(m, n)
    A = LLL.reduction(A)
    M = GSO.Mat(A)
    M.update_gso()
    t = list(make_integer_matrix(n, n)[0])
    with pytest.raises(NotImplementedError):
        CVP.closest_vector(A, t)
