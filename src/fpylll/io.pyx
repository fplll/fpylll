# -*- coding: utf-8 -*-
include "fpylll/config.pxi"

from cpython.int cimport PyInt_AS_LONG
from fpylll.gmp.mpz cimport mpz_init, mpz_clear, mpz_set
from fpylll.gmp.pylong cimport mpz_get_pyintlong, mpz_set_pylong
from .gmp.mpz cimport mpz_t, mpz_set_si, mpz_set
from cpython.version cimport PY_MAJOR_VERSION

try:
    from sage.all import ZZ
    have_sage = True
except Exception:
    have_sage = False

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1:
    """
    Assign Python integer to Z_NR[mpz_t]
    """
    cdef mpz_t tmp
    mpz_init(tmp)
    try:
        assign_mpz(tmp, value)
        t.set(tmp)
    finally:
        mpz_clear(tmp)

cdef int assign_mpz(mpz_t& t, value) except -1:
    """
    Assign Python integer to Z_NR[mpz_t]
    """
    if isinstance(value, int) and PY_MAJOR_VERSION == 2:
            mpz_set_si(t, PyInt_AS_LONG(value))
            return 0
    if isinstance(value, long):
        mpz_set_pylong(t, value)
        return 0
    if have_sage:
        from sage.rings.integer import Integer
        if isinstance(value, Integer):
            value = long(value)
            mpz_set_pylong(t, value)
            return 0

    raise NotImplementedError("Type '%s' not supported"%type(value))

cdef object mpz_get_python(mpz_srcptr z):
    r = mpz_get_pyintlong(z)
    if have_sage:
        from sage.rings.integer import Integer
        return Integer(r)
    else:
        return r
