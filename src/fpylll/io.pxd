include "fpylll/config.pxi"

from .fplll.fplll cimport Z_NR
from .gmp.mpz cimport mpz_t
from .gmp.types cimport mpz_srcptr
from .fplll.decl cimport fp_nr_t, vector_fp_nr_t, vector_z_nr_t
from fplll.fplll cimport FloatType, IntType

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1
cdef int assign_mpz(mpz_t& t, value) except -1
cdef object mpz_get_python(mpz_srcptr z)
cdef void vector_fp_nr_barf(vector_fp_nr_t &out, object inp, FloatType float_type)
cdef object vector_fp_nr_slurp(vector_fp_nr_t &inp, FloatType float_type)
cdef object vector_z_nr_slurp(vector_z_nr_t &inp, IntType int_type)
