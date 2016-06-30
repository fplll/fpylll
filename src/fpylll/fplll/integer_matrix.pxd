# -*- coding: utf-8 -*-

from fpylll.gmp.types cimport mpz_t
from fplll cimport ZZ_mat

cdef class IntegerMatrix:
    cdef ZZ_mat[mpz_t]  *_core

cdef class IntegerMatrixRow:
    cdef int row
    cdef IntegerMatrix m
