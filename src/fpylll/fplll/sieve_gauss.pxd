# -*- coding: utf-8 -*-

from fpylll.gmp.types cimport mpz_t
from .fplll cimport FP_NR, IntType
from .decl cimport gauss_sieve_core_t

cdef class GaussSieve:
    cdef IntType _type
    cdef gauss_sieve_core_t _core
