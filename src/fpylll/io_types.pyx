# -*- coding: utf-8 -*-
include "config.pxi"
include "cysignals/signals.pxi"

from cpython.int cimport PyInt_AS_LONG
from fpylll.gmp.mpz cimport mpz_init, mpz_clear, mpz_set_si
from fpylll.gmp.pylong cimport mpz_get_pyintlong, mpz_set_pylong

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1:
    """
    Assign Python integer to Z_NR[mpz_t]
    """
    cdef mpz_t tmp
    mpz_init(tmp)
    if isinstance(value, int):
        mpz_set_si(tmp, PyInt_AS_LONG(value))
    elif isinstance(value, long):
        mpz_set_pylong(tmp, value)
    else:
        mpz_clear(tmp)
        msg = "Only Python ints and longs are currently supported, but got type '%s'"%type(value)
        raise NotImplementedError(msg)

    t.set(tmp)
    mpz_clear(tmp)


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
