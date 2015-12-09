from gmp.mpz cimport mpz_t
from fplll cimport FloatType, Z_NR
from fplll cimport BKZParam as BKZParam_c

cdef FloatType check_float_type(object float_type)
cdef int preprocess_indices(int &i, int &j, int m, int n) except -1
cdef int check_precision(int precision) except -1
cdef int check_eta(float eta) except -1
cdef int check_delta(float delta) except -1
cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1
cdef int assign_mpz(mpz_t& t, value) except -1
