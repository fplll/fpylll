# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, set_random_seed


def make_integer_matrix(m, n):
    A = IntegerMatrix(m, n)
    A.randomize("uniform", bits=m+n)
    return A


def test_randomize():
    set_random_seed(1337)
    A0 = make_integer_matrix(20, 20)
    set_random_seed(1337)
    A1 = make_integer_matrix(20, 20)
    for i in range(20):
        for j in range(20):
            assert A0[i, j] == A1[i, j]
