# -*- coding: utf-8 -*-

from fpylll.gmp.types cimport mpz_t
from .fplll cimport IntType
from .decl cimport zz_mat_core_t

cdef class IntegerMatrix:
    cdef IntType _type
    cdef zz_mat_core_t _core
    cdef long _nrows(self)
    cdef long _ncols(self)
    cdef object _get(self, int i, int j)
    cdef int _set(self, int i, int j, value) except -1

cdef class IntegerMatrixRow:
    cdef int row
    cdef IntegerMatrix m
