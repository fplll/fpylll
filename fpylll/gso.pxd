# -*- coding: utf-8 -*-

from gmp.mpz cimport mpz_t
from mpfr.mpfr cimport mpfr_t
from integer_matrix cimport IntegerMatrix
from fplll cimport MatGSO as MatGSO_c, Z_NR, FP_NR, Matrix

cdef class MatGSO:
    """
    """
    cdef MatGSO_c[Z_NR[mpz_t], FP_NR[double]] *_core_mpz_double;
    cdef MatGSO_c[Z_NR[mpz_t], FP_NR[mpfr_t]] *_core_mpz_mpfr;
    cdef IntegerMatrix _B
    cdef IntegerMatrix _U
    cdef IntegerMatrix _UinvT
