# -*- coding: utf-8 -*-

from fpylll import GSO, IntegerMatrix, LLL, SVP, Enumeration
import pytest

dimensions = ((3, 3), (10, 10), (20, 20), (30, 30), (40, 40))


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
        E = Enumeration(M)
        _, v1 = E.enumerate(0, M.d, M.get_r(0, 0), 0)[0]
        v1 = A.multiply_left(v1)
        nv1 = sum([v_**2 for v_ in v1])

        v0 = SVP.shortest_vector(A)
        nv0 = sum([v_**2 for v_ in v0])

        assert nv0 == nv1


def test_svp_params():
    params = [{"pruning": False, "preprocess": 2},
              {"pruning": True, "preprocess": 30},
              {"method": "proved"},
              {"max_aux_solutions": 20}]

    for kwds in params:
        for m, n in dimensions:
            A = make_integer_matrix(m, n)
            SVP.shortest_vector(A, **kwds)


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
