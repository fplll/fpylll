# copied/adapted from Sage development tree version 6.9
"""
Various functions to deal with conversion mpz <-> Python int/long
"""

cdef extern from "Python.h":
    cdef _PyLong_New(Py_ssize_t s)
    cdef long PyLong_SHIFT
    ctypedef unsigned int digit
    ctypedef struct PyLongObject:
        digit* ob_digit

    ctypedef struct PyObject:
        pass

    ctypedef struct PyVarObject:
        PyObject ob_base
        Py_ssize_t ob_size

from fpylll.gmp.types cimport *

cdef mpz_get_pylong(mpz_srcptr z)
cdef mpz_get_pyintlong(mpz_srcptr z)
cdef int mpz_set_pylong(mpz_ptr z, L) except -1
cdef Py_hash_t mpz_pythonhash(mpz_srcptr z)
