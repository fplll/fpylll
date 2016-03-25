include "config.pxi"

from fplll cimport Z_NR
from gmp.mpz cimport mpz_t
from gmp.types cimport mpz_srcptr
from fpylll.gmp.pylong cimport mpz_get_pyintlong

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1
cdef int assign_mpz(mpz_t& t, value) except -1

IF HAVE_SAGE:
    raise NotImplementedError("Sage support not implemented yet.")
ELSE:
    from fpylll.gmp.pylong cimport mpz_get_pyintlong as mpz_get_python
