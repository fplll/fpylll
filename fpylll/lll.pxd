# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

from gmp.mpz cimport mpz_t
from mpfr.mpfr cimport mpfr_t
from fplll cimport Z_NR, FP_NR
from fplll cimport LLLReduction as LLLReduction_c
from gso cimport MatGSO

cdef class LLLReduction:
    cdef LLLReduction_c[Z_NR[mpz_t], FP_NR[double]] *_core_mpz_double
    cdef LLLReduction_c[Z_NR[mpz_t], FP_NR[mpfr_t]] *_core_mpz_mpfr
    cdef MatGSO m
    cdef double _delta
    cdef double _eta
