# -*- coding: utf-8 -*-

from .fplll cimport Wrapper as Wrapper_c
from .integer_matrix cimport IntegerMatrix

cdef class Wrapper:
    cdef Wrapper_c *_core
    cdef object _called

    cdef readonly IntegerMatrix B
    cdef readonly IntegerMatrix U
    cdef readonly IntegerMatrix UinvT
