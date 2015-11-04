# distutils: language = c++
# distutils: libraries = gmp mpfr fplll qd

from gmp.mpz cimport mpz_t
from mpfr.mpfr cimport mpfr_t
from qd.qd cimport dd_real, qd_real
from fplll cimport Z_NR, FP_NR
from fplll cimport MatGSO, LLLReduction, BKZAutoAbort

ctypedef enum fplll_type:
    mpz_double =  1
    mpz_ld     =  2
    mpz_dd     =  4
    mpz_qd     =  8
    mpz_dpe    = 16
    mpz_mpfr   = 32

ctypedef union mat_gso_core:
    MatGSO[Z_NR[mpz_t], FP_NR[double]] *mpz_double
    # MatGSO[Z_NR[mpz_t], FP_NR[longdouble]] *mpz_ld
    MatGSO[Z_NR[mpz_t], FP_NR[dd_real]] *mpz_dd
    MatGSO[Z_NR[mpz_t], FP_NR[qd_real]] *mpz_qd
    #MatGSO[Z_NR[mpz_t], FP_NR[dpe]] *mpz_dpe
    MatGSO[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr

ctypedef union lll_reduction_core:
    LLLReduction[Z_NR[mpz_t], FP_NR[double]] *mpz_double
    #LLLReduction[Z_NR[mpz_t], FP_NR[longdouble]] *mpz_ld
    LLLReduction[Z_NR[mpz_t], FP_NR[dd_real]] *mpz_dd
    LLLReduction[Z_NR[mpz_t], FP_NR[qd_real]] *mpz_qd
    #LLLReduction[Z_NR[mpz_t], FP_NR[dpe]] *mpz_dpe
    LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr

ctypedef union bkz_auto_abort_core:
    BKZAutoAbort[FP_NR[double]] *mpz_double
    #BKZAutoAbort[FP_NR[ld]] *mpz_ld
    BKZAutoAbort[FP_NR[dd_real]] *mpz_dd
    BKZAutoAbort[FP_NR[qd_real]] *mpz_qd
    #BKZAutoAbort[FP_NR[dpe]] *mpz_dpe
    BKZAutoAbort[FP_NR[mpfr_t]] *mpz_mpfr
