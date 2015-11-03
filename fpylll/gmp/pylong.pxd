# copied/adapted from Sage development tree version 6.9
"""
Various functions to deal with conversion mpz <-> Python int/long
"""

cdef extern from "longintrepr.h":
    cdef _PyLong_New(Py_ssize_t s)
    cdef long PyLong_SHIFT
    ctypedef unsigned int digit
    ctypedef struct PyLongObject:
        Py_ssize_t ob_size
        digit* ob_digit

from fpylll.gmp.types cimport *

cdef mpz_get_pylong(mpz_srcptr z)
cdef mpz_get_pyintlong(mpz_srcptr z)
cdef int mpz_set_pylong(mpz_ptr z, L) except -1
cdef Py_hash_t mpz_pythonhash(mpz_srcptr z)
