# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

from gmp.mpz cimport mpz_t
from mpfr.mpfr cimport mpfr_t
from fplll cimport Z_NR, FP_NR
from fplll cimport MatGSO, LLLReduction, BKZAutoAbort

ctypedef enum fplll_type:
    mpz_double = 1
    mpz_mpfr   = 2

ctypedef union mat_gso_core:
    MatGSO[Z_NR[mpz_t], FP_NR[double]] *mpz_double
    MatGSO[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr

ctypedef union lll_reduction_core:
    LLLReduction[Z_NR[mpz_t], FP_NR[double]] *mpz_double
    LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr

ctypedef union bkz_auto_abort_core:
    BKZAutoAbort[FP_NR[double]] *mpz_double
    BKZAutoAbort[FP_NR[mpfr_t]] *mpz_mpfr
