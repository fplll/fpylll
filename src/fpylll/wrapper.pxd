# -*- coding: utf-8 -*-

from fplll cimport Wrapper as Wrapper_c
from integer_matrix cimport IntegerMatrix

cdef class Wrapper:
    cdef Wrapper_c *_core
    cdef object _called

    cdef IntegerMatrix _B
    cdef IntegerMatrix _U
    cdef IntegerMatrix _UinvT
