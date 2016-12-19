# -*- coding: utf-8 -*-
include "fpylll/config.pxi"

"""
Fpylll datatypes

These are mainly for internal consumption
"""
from fpylll.gmp.mpz cimport mpz_t
from fpylll.mpfr.mpfr cimport mpfr_t

IF HAVE_QD:
    from fpylll.qd.qd cimport dd_real, qd_real

from fplll cimport dpe_t
from fplll cimport Z_NR, FP_NR
from fplll cimport MatGSO, LLLReduction, BKZAutoAbort, BKZReduction, Enumeration, FastEvaluator, FastErrorBoundedEvaluator

from libcpp.vector cimport vector

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

IF HAVE_LONG_DOUBLE:
    IF HAVE_QD:
        # we cannot use a union because of non-trivial constructors
        ctypedef struct fp_nr_t:
            FP_NR[double] double
            FP_NR[longdouble] ld
            FP_NR[dpe_t] dpe
            FP_NR[dd_real] dd
            FP_NR[qd_real] qd
            FP_NR[mpfr_t] mpfr
    ELSE:
        ctypedef struct fp_nr_t:
            FP_NR[double] double
            FP_NR[longdouble] ld
            FP_NR[dpe_t] dpe
            FP_NR[mpfr_t] mpfr

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
        ctypedef union bkz_reduction_core_t:
            BKZReduction[FP_NR[double]] *mpz_double
            BKZReduction[FP_NR[longdouble]] *mpz_ld
            BKZReduction[FP_NR[dpe_t]] *mpz_dpe
            BKZReduction[FP_NR[dd_real]] *mpz_dd
            BKZReduction[FP_NR[qd_real]] *mpz_qd
            BKZReduction[FP_NR[mpfr_t]] *mpz_mpfr
    ELSE:
        ctypedef union bkz_reduction_core_t:
            BKZReduction[FP_NR[double]] *mpz_double
            BKZReduction[FP_NR[longdouble]] *mpz_ld
            BKZReduction[FP_NR[dpe_t]] *mpz_dpe
            BKZReduction[FP_NR[mpfr_t]] *mpz_mpfr

    IF HAVE_QD:
        ctypedef union fast_evaluator_core_t:
            FastEvaluator[FP_NR[double]] *double
            FastEvaluator[FP_NR[longdouble]] *ld
            FastEvaluator[FP_NR[dpe_t]] *dpe
            FastEvaluator[FP_NR[dd_real]] *dd
            FastEvaluator[FP_NR[qd_real]] *qd
            FastErrorBoundedEvaluator *mpfr
    ELSE:
        ctypedef union fast_evaluator_core_t:
            FastEvaluator[FP_NR[double]] *double
            FastEvaluator[FP_NR[longdouble]] *ld
            FastEvaluator[FP_NR[dpe_t]] *dpe
            FastErrorBoundedEvaluator *mpfr

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

    IF HAVE_QD:
        # we cannot use a union because of non-trivial constructors
        ctypedef struct vector_fp_nr_t:
            vector[FP_NR[double]] double
            vector[FP_NR[longdouble]] ld
            vector[FP_NR[dpe_t]] dpe
            vector[FP_NR[dd_real]] dd
            vector[FP_NR[qd_real]] qd
            vector[FP_NR[mpfr_t]] mpfr
    ELSE:
        ctypedef struct vector_fp_nr_t:
            vector[FP_NR[double]] double
            vector[FP_NR[longdouble]] ld
            vector[FP_NR[dpe_t]] dpe
            vector[FP_NR[mpfr_t]] mpfr
ELSE:
    IF HAVE_QD:
        # we cannot use a union because of non-trivial constructors
        ctypedef struct fp_nr_t:
            FP_NR[double] double
            FP_NR[dpe_t] dpe
            FP_NR[dd_real] dd
            FP_NR[qd_real] qd
            FP_NR[mpfr_t] mpfr
    ELSE:
        ctypedef struct fp_nr_t:
            FP_NR[double] double
            FP_NR[dpe_t] dpe
            FP_NR[mpfr_t] mpfr

    IF HAVE_QD:
        ctypedef union mat_gso_core_t:
            MatGSO[Z_NR[mpz_t], FP_NR[double]] *mpz_double
            MatGSO[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            MatGSO[Z_NR[mpz_t], FP_NR[dd_real]] *mpz_dd
            MatGSO[Z_NR[mpz_t], FP_NR[qd_real]] *mpz_qd
            MatGSO[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
    ELSE:
        ctypedef union mat_gso_core_t:
            MatGSO[Z_NR[mpz_t], FP_NR[double]] *mpz_double
            MatGSO[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            MatGSO[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr

    IF HAVE_QD:
        ctypedef union lll_reduction_core_t:
            LLLReduction[Z_NR[mpz_t], FP_NR[double]] *mpz_double
            LLLReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            LLLReduction[Z_NR[mpz_t], FP_NR[dd_real]] *mpz_dd
            LLLReduction[Z_NR[mpz_t], FP_NR[qd_real]] *mpz_qd
            LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
    ELSE:
        ctypedef union lll_reduction_core_t:
            LLLReduction[Z_NR[mpz_t], FP_NR[double]] *mpz_double
            LLLReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr

    IF HAVE_QD:
        ctypedef union bkz_auto_abort_core_t:
            BKZAutoAbort[FP_NR[double]] *mpz_double
            BKZAutoAbort[FP_NR[dpe_t]] *mpz_dpe
            BKZAutoAbort[FP_NR[dd_real]] *mpz_dd
            BKZAutoAbort[FP_NR[qd_real]] *mpz_qd
            BKZAutoAbort[FP_NR[mpfr_t]] *mpz_mpfr
    ELSE:
        ctypedef union bkz_auto_abort_core_t:
            BKZAutoAbort[FP_NR[double]] *mpz_double
            BKZAutoAbort[FP_NR[dpe_t]] *mpz_dpe
            BKZAutoAbort[FP_NR[mpfr_t]] *mpz_mpfr

    IF HAVE_QD:
        ctypedef union bkz_reduction_core_t:
            BKZReduction[FP_NR[double]] *mpz_double
            BKZReduction[FP_NR[dpe_t]] *mpz_dpe
            BKZReduction[FP_NR[dd_real]] *mpz_dd
            BKZReduction[FP_NR[qd_real]] *mpz_qd
            BKZReduction[FP_NR[mpfr_t]] *mpz_mpfr
    ELSE:
        ctypedef union bkz_reduction_core_t:
            BKZReduction[FP_NR[double]] *mpz_double
            BKZReduction[FP_NR[dpe_t]] *mpz_dpe
            BKZReduction[FP_NR[mpfr_t]] *mpz_mpfr

    IF HAVE_QD:
        ctypedef union fast_evaluator_core_t:
            FastEvaluator[FP_NR[double]] *double
            FastEvaluator[FP_NR[dpe_t]] *dpe
            FastEvaluator[FP_NR[dd_real]] *dd
            FastEvaluator[FP_NR[qd_real]] *qd
            FastEvaluator[FP_NR[mpfr_t]] *mpfr
    ELSE:
        ctypedef union fast_evaluator_core_t:
            FastEvaluator[FP_NR[double]] *double
            FastEvaluator[FP_NR[dpe_t]] *dpe
            FastEvaluator[FP_NR[mpfr_t]] *mpfr

    IF HAVE_QD:
        ctypedef union enumeration_core_t:
            Enumeration[FP_NR[double]] *double
            Enumeration[FP_NR[dpe_t]] *dpe
            Enumeration[FP_NR[dd_real]] *dd
            Enumeration[FP_NR[qd_real]] *qd
            Enumeration[FP_NR[mpfr_t]] *mpfr
    ELSE:
        ctypedef union enumeration_core_t:
            Enumeration[FP_NR[double]] *double
            Enumeration[FP_NR[dpe_t]] *dpe
            Enumeration[FP_NR[mpfr_t]] *mpfr

    IF HAVE_QD:
        # we cannot use a union because of non-trivial constructors
        ctypedef struct vector_fp_nr_t:
            vector[FP_NR[double]] double
            vector[FP_NR[dpe_t]] dpe
            vector[FP_NR[dd_real]] dd
            vector[FP_NR[qd_real]] qd
            vector[FP_NR[mpfr_t]] mpfr
    ELSE:
        ctypedef struct vector_fp_nr_t:
            vector[FP_NR[double]] double
            vector[FP_NR[dpe_t]] dpe
            vector[FP_NR[mpfr_t]] mpfr
