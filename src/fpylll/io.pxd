include "fpylll/config.pxi"

from .fplll.fplll cimport Z_NR
from .gmp.mpz cimport mpz_t
from .gmp.types cimport mpz_srcptr

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1
cdef int assign_mpz(mpz_t& t, value) except -1
cdef object mpz_get_python(mpz_srcptr z)
