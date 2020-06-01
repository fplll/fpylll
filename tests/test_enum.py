# -*- coding: utf-8 -*-
from fpylll import IntegerMatrix, GSO, LLL, Enumeration
from fpylll.config import float_types, int_types
from copy import copy


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
