from gmp.mpz cimport mpz_t
from fplll.fplll cimport FloatType, Z_NR
from fplll.fplll cimport BKZParam as BKZParam_c

cdef FloatType check_float_type(object float_type)
cdef int preprocess_indices(int &i, int &j, int m, int n) except -1
cdef int check_precision(int precision) except -1
cdef int check_eta(float eta) except -1
cdef int check_delta(float delta) except -1
cdef int check_descent_method(object descent_method) except -1
