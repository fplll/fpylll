# -*- coding: utf-8 -*-

from fpylll.gmp.types cimport mpz_t
from fplll cimport FP_NR
from fplll cimport GaussSieve as GaussSieve_c

cdef class GaussSieve:
    cdef GaussSieve_c[mpz_t, FP_NR[double]]  *_core
