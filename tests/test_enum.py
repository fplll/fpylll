# -*- coding: utf-8 -*-
from fpylll import IntegerMatrix, GSO, LLL, Enumeration
from fpylll.config import float_types, int_types
from copy import copy

import tools


def make_integer_matrix(m, n, int_type="mpz"):
    A = IntegerMatrix(m, n, int_type=int_type)
    A.randomize("qary", k=m // 2, bits=m)
    return A


def test_enum_init():
    for int_type in int_types:
        A = make_integer_matrix(20, 20, int_type=int_type)
        for float_type in float_types:
            M = GSO.Mat(copy(A), float_type=float_type)
            enum_obj = Enumeration(M)
            del enum_obj


def test_enum_enum():
    for int_type in int_types:
        A = make_integer_matrix(20, 20, int_type=int_type)
        LLL.reduction(A)
        for float_type in float_types:
            M = GSO.Mat(copy(A), float_type=float_type)
            M.update_gso()
            enum_obj = Enumeration(M)
            enum_obj.enumerate(0, M.d, M.get_r(0, 0), 0)


def test_enum_gram_coherence():
    """
        Test if the enumeration algorithm is consistent with the Gram matrices
        The vectors returned by the enumeration should be the same wether a
        lattice is given by its basis or by its Gram matrix
    """

    dimensions = ((3, 3), (10, 10), (20, 20), (25, 25))

    for m, n in dimensions:
        for int_type in int_types:
            A = make_integer_matrix(m, n, int_type=int_type)
            LLL.reduction(A)
            G = tools.compute_gram(A)
            for float_type in float_types:
                M_A = GSO.Mat(copy(A), float_type=float_type, gram=False)
                M_G = GSO.Mat(copy(G), float_type=float_type, gram=True)

                M_A.update_gso()
                M_G.update_gso()

                enum_obj_a = Enumeration(M_A, nr_solutions=min(m, 5))
                shortest_vectors_a = enum_obj_a.enumerate(0, M_A.d, M_A.get_r(0, 0), 0)

                enum_obj_g = Enumeration(M_G, nr_solutions=min(m, 5))
                shortest_vectors_g = enum_obj_g.enumerate(0, M_G.d, M_G.get_r(0, 0), 0)

                for i in range(len(shortest_vectors_a)):
                    assert shortest_vectors_a[i] == shortest_vectors_g[i]
