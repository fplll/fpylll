include "fpylll/config.pxi"

from cpython.int cimport PyInt_AS_LONG
from fplll.fplll cimport Z_NR
from gmp.mpz cimport mpz_t, mpz_set_si, mpz_set
from gmp.types cimport mpz_srcptr
from fpylll.gmp.pylong cimport mpz_get_pyintlong, mpz_set_pylong

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1

IF HAVE_SAGE:
    from sage.rings.integer cimport Integer
    cdef Integer mpz_get_python(mpz_srcptr z)
ELSE:
    from fpylll.gmp.pylong cimport mpz_get_pyintlong as mpz_get_python

cdef inline int assign_mpz(mpz_t& t, value) except -1:
    """
    Assign Python integer to Z_NR[mpz_t]
    """
    if isinstance(value, int):
        mpz_set_si(t, PyInt_AS_LONG(value))
        return 0
    if isinstance(value, long):
        mpz_set_pylong(t, value)
        return 0
    IF HAVE_SAGE:
        if isinstance(value, Integer):
            mpz_set(t, (<Integer>value).value)
            return 0

    raise NotImplementedError("Type '%s' not supported"%type(value))
