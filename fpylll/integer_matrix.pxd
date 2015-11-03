# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

from gmp.types cimport mpz_t
from fplll cimport ZZ_mat

cdef class IntegerMatrix:
    cdef ZZ_mat[mpz_t]  *_core
