# -*- coding: utf-8 -*-
include "config.pxi"
include "cysignals/signals.pxi"

from cpython.int cimport PyInt_AS_LONG
from fpylll.gmp.mpz cimport mpz_init, mpz_clear, mpz_set_si, mpz_set
from fpylll.gmp.pylong cimport mpz_get_pyintlong, mpz_set_pylong

IF HAVE_SAGE:
    from sage.rings.integer cimport Integer
    from sage.ext.stdsage cimport PY_NEW

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1:
    """
    Assign Python integer to Z_NR[mpz_t]
    """
    cdef mpz_t tmp
    mpz_init(tmp)
    if isinstance(value, int):
        mpz_set_si(tmp, PyInt_AS_LONG(value))
        t.set(tmp)
        mpz_clear(tmp)
        return 0
    if isinstance(value, long):
        mpz_set_pylong(tmp, value)
        t.set(tmp)
        mpz_clear(tmp)
        return 0
    IF HAVE_SAGE:
        if isinstance(value, Integer):
            mpz_set(tmp, (<Integer>value).value)
            t.set(tmp)
            mpz_clear(tmp)
            return 0

    mpz_clear(tmp)
    raise NotImplementedError("Type '%s' not supported"%type(value))


cdef int assign_mpz(mpz_t& t, value) except -1:
    """
    Assign Python integer to Z_NR[mpz_t]
    """
    if isinstance(value, int):
        mpz_set_si(t, PyInt_AS_LONG(value))
        return 0
    if isinstance(value, long):
        mpz_set_pylong(t, value)
        return 0

    msg = "Only Python ints and longs are currently supported, but got type '%s'"%type(value)
    raise NotImplementedError(msg)


IF HAVE_SAGE:
    cdef Integer mpz_get_python(mpz_srcptr z):
        cdef Integer x = PY_NEW(Integer)
        mpz_set(x.value, z)
        return x
