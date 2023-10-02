# copied/adapted from Sage development tree version 6.9
"""
Various functions to deal with conversion mpz <-> Python int/long
"""

from cpython.longintrepr cimport py_long
from fpylll.gmp.types cimport *

cdef mpz_get_pylong(mpz_srcptr z)
cdef mpz_get_pyintlong(mpz_srcptr z)
cdef int mpz_set_pylong(mpz_ptr z, py_long L) except -1
cdef Py_hash_t mpz_pythonhash(mpz_srcptr z)
