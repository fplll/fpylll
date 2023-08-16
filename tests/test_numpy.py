# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, GSO

try:
    from fpylll.numpy import dump_mu, dump_r
    have_numpy = True
except ImportError:
    have_numpy = False


def test_dump_mu(nrows=10):
    A = IntegerMatrix(nrows, nrows)
    A.randomize("ntrulike", bits=10, q=1023)
    M  = GSO.Mat(A)
    if not have_numpy:
        return

    M.update_gso()
    mu = dump_mu(M, 0, nrows)

    for i in range(nrows):
        for j in range(i):
            assert abs(M.get_mu(i, j) - mu[i, j]) < 0.001


def test_dump_r(nrows=10):
    A = IntegerMatrix(nrows, nrows)
    A.randomize("ntrulike", bits=10, q=1023)
    M  = GSO.Mat(A)
    if not have_numpy:
        return

    M.update_gso()
    r = dump_r(M, 0, nrows)

    for i in range(nrows):
        assert abs(M.get_r(i, i) - r[i]) < 0.001


def test_is_numpy_integer(nrows=10):
    if not have_numpy:
        return

    import numpy as np
    B = np.eye(nrows, dtype=np.int32)
    Bfpy = IntegerMatrix.from_matrix(B)
    for i in range(nrows):
        assert Bfpy[i][i] == 1
