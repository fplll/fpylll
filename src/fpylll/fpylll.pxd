# -*- coding: utf-8 -*-
include "config.pxi"

from gmp.mpz cimport mpz_t
from mpfr.mpfr cimport mpfr_t

IF HAVE_QD:
    from qd.qd cimport dd_real, qd_real

from fplll cimport dpe_t
from fplll cimport Z_NR, FP_NR
from fplll cimport MatGSO, LLLReduction, BKZAutoAbort, Enumeration, FastEvaluator


IF HAVE_QD:
    ctypedef enum fplll_type_t:
        mpz_double =  1
        mpz_ld     =  2
        mpz_dpe    =  4
        mpz_dd     =  8
        mpz_qd     = 16
        mpz_mpfr   = 32
ELSE:
    ctypedef enum fplll_type_t:
        mpz_double =  1
        mpz_ld     =  2
        mpz_dpe    =  4


        mpz_mpfr   = 32

IF HAVE_QD:
    ctypedef union mat_gso_core_t:
        MatGSO[Z_NR[mpz_t], FP_NR[double]] *mpz_double
        MatGSO[Z_NR[mpz_t], FP_NR[longdouble]] *mpz_ld
        MatGSO[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
        MatGSO[Z_NR[mpz_t], FP_NR[dd_real]] *mpz_dd
        MatGSO[Z_NR[mpz_t], FP_NR[qd_real]] *mpz_qd
        MatGSO[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
ELSE:
    ctypedef union mat_gso_core_t:
        MatGSO[Z_NR[mpz_t], FP_NR[double]] *mpz_double
        MatGSO[Z_NR[mpz_t], FP_NR[longdouble]] *mpz_ld
        MatGSO[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
        MatGSO[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr

IF HAVE_QD:
    ctypedef union lll_reduction_core_t:
        LLLReduction[Z_NR[mpz_t], FP_NR[double]] *mpz_double
        LLLReduction[Z_NR[mpz_t], FP_NR[longdouble]] *mpz_ld
        LLLReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
        LLLReduction[Z_NR[mpz_t], FP_NR[dd_real]] *mpz_dd
        LLLReduction[Z_NR[mpz_t], FP_NR[qd_real]] *mpz_qd
        LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
ELSE:
    ctypedef union lll_reduction_core_t:
        LLLReduction[Z_NR[mpz_t], FP_NR[double]] *mpz_double
        LLLReduction[Z_NR[mpz_t], FP_NR[longdouble]] *mpz_ld
        LLLReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
        LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr

IF HAVE_QD:
    ctypedef union bkz_auto_abort_core_t:
        BKZAutoAbort[FP_NR[double]] *mpz_double
        BKZAutoAbort[FP_NR[longdouble]] *mpz_ld
        BKZAutoAbort[FP_NR[dpe_t]] *mpz_dpe
        BKZAutoAbort[FP_NR[dd_real]] *mpz_dd
        BKZAutoAbort[FP_NR[qd_real]] *mpz_qd
        BKZAutoAbort[FP_NR[mpfr_t]] *mpz_mpfr
ELSE:
    ctypedef union bkz_auto_abort_core_t:
        BKZAutoAbort[FP_NR[double]] *mpz_double
        BKZAutoAbort[FP_NR[longdouble]] *mpz_ld
        BKZAutoAbort[FP_NR[dpe_t]] *mpz_dpe
        BKZAutoAbort[FP_NR[mpfr_t]] *mpz_mpfr

IF HAVE_QD:
    ctypedef union fast_evaluator_core_t:
        FastEvaluator[FP_NR[double]] *double
        FastEvaluator[FP_NR[longdouble]] *ld
        FastEvaluator[FP_NR[dpe_t]] *dpe
        FastEvaluator[FP_NR[dd_real]] *dd
        FastEvaluator[FP_NR[qd_real]] *qd
        FastEvaluator[FP_NR[mpfr_t]] *mpfr
ELSE:
    ctypedef union fast_evaluator_core_t:
        FastEvaluator[FP_NR[double]] *double
        FastEvaluator[FP_NR[longdouble]] *ld
        FastEvaluator[FP_NR[dpe_t]] *dpe
        FastEvaluator[FP_NR[mpfr_t]] *mpfr

IF HAVE_QD:
    ctypedef union enumeration_core_t:
        Enumeration[FP_NR[double]] *double
        Enumeration[FP_NR[longdouble]] *ld
        Enumeration[FP_NR[dpe_t]] *dpe
        Enumeration[FP_NR[dd_real]] *dd
        Enumeration[FP_NR[qd_real]] *qd
        Enumeration[FP_NR[mpfr_t]] *mpfr
ELSE:
    ctypedef union enumeration_core_t:
        Enumeration[FP_NR[double]] *double
        Enumeration[FP_NR[longdouble]] *ld
        Enumeration[FP_NR[dpe_t]] *dpe
        Enumeration[FP_NR[mpfr_t]] *mpfr
