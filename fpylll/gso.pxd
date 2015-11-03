# -*- coding: utf-8 -*-

from integer_matrix cimport IntegerMatrix
from fpylll cimport mat_gso_core, fplll_type

cdef class MatGSO:
    cdef fplll_type _type
    cdef mat_gso_core _core

    cdef IntegerMatrix _B
    cdef IntegerMatrix _U
    cdef IntegerMatrix _UinvT
