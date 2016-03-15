# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, GSO

try:
    import numpy
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

    mu = numpy.ndarray(dtype='double', shape=(nrows, nrows))

    M.update_gso()
    dump_mu(mu, M, 0, 10)

    for i in range(nrows):
        for j in range(nrows):
            assert abs(M.get_mu(i, j) - mu[i, j]) < 0.001


def test_dump_r(nrows=10):
    A = IntegerMatrix(nrows, nrows)
    A.randomize("ntrulike", bits=10, q=1023)
    M  = GSO.Mat(A)
    if not have_numpy:
        return

    r = numpy.ndarray(dtype='double', shape=nrows)

    M.update_gso()
    dump_r(r, M, 0, 10)

    for i in range(nrows):
        assert abs(M.get_r(i, i) - r[i]) < 0.001
