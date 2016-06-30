# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"

from fpylll.gmp.mpz cimport mpz_init, mpz_clear, mpz_set

IF HAVE_SAGE:
    from sage.ext.stdsage cimport PY_NEW

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


IF HAVE_SAGE:
    cdef Integer mpz_get_python(mpz_srcptr z):
        cdef Integer x = PY_NEW(Integer)
        mpz_set(x.value, z)
        return x
