# -*- coding: utf-8 -*-

from cysignals.signals import SignalError
from pytest import raises

from fpylll import IntegerMatrix, FPLLL


def make_integer_matrix(m, n, int_type="mpz"):
    A = IntegerMatrix(m, n, int_type=int_type)
    A.randomize("qary", k=m//2, bits=m)
    return A


def test_zero_bits():
    with raises(SignalError):
        IntegerMatrix.random(10, "qary", k=5, bits=0)


def test_randomize():
    FPLLL.set_random_seed(1337)
    A0 = make_integer_matrix(20, 20)
    FPLLL.set_random_seed(1337)
    A1 = make_integer_matrix(20, 20)
    for i in range(20):
        for j in range(20):
            assert A0[i, j] == A1[i, j]
