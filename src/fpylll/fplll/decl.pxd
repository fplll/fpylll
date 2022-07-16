# -*- coding: utf-8 -*-
"""
Fpylll datatypes

These are mainly for internal consumption
"""

include "fpylll/config.pxi"

from fpylll.gmp.mpz cimport mpz_t
from fpylll.mpfr.mpfr cimport mpfr_t

IF HAVE_QD:
    from fpylll.qd.qd cimport dd_real, qd_real

from .fplll cimport dpe_t
from .fplll cimport Z_NR, FP_NR
from .fplll cimport ZZ_mat, MatGSOInterface, LLLReduction, BKZAutoAbort, BKZReduction, Enumeration
from .fplll cimport Evaluator, FastEvaluator, ErrorBoundedEvaluator, FastErrorBoundedEvaluator, Pruner

from libcpp.vector cimport vector

ctypedef double d_t

IF HAVE_LONG_DOUBLE:
    ctypedef long double ld_t

IF HAVE_QD:
    ctypedef dd_real dd_t
    ctypedef qd_real qd_t

ctypedef enum fplll_mat_gso_alg_type_t:
    mat_gso_gso_t  = 0
    mat_gso_gram_t = 1

IF HAVE_QD:
    ctypedef enum fplll_mat_gso_data_type_t:
        mat_gso_mpz_d      =    1
        mat_gso_mpz_ld     =    2
        mat_gso_mpz_dpe    =    4
        mat_gso_mpz_dd     =    8
        mat_gso_mpz_qd     =   16
        mat_gso_mpz_mpfr   =   32
        mat_gso_long_d     =   64
        mat_gso_long_ld    =  128
        mat_gso_long_dpe   =  256
        mat_gso_long_dd    =  512
        mat_gso_long_qd    = 1024
        mat_gso_long_mpfr  = 2048

ELSE:
    ctypedef enum fplll_mat_gso_data_type_t:
        mat_gso_mpz_d      =    1
        mat_gso_mpz_ld     =    2
        mat_gso_mpz_dpe    =    4
        mat_gso_mpz_mpfr   =   32
        mat_gso_long_d     =   64
        mat_gso_long_ld    =  128
        mat_gso_long_dpe   =  256
        mat_gso_long_mpfr  = 2048

IF HAVE_QD:
    ctypedef enum fplll_nr_type_t:
        nr_d      =  1
        nr_ld     =  2
        nr_dpe    =  4
        nr_dd     =  8
        nr_qd     = 16
        nr_mpfr   = 32
ELSE:
    ctypedef enum fplll_nr_type_t:
        nr_d      =  1
        nr_ld     =  2
        nr_dpe    =  4
        nr_mpfr   = 32

ctypedef enum fplll_z_type_t:
    z_long    =  1
    z_mpz     =  2

ctypedef union zz_mat_core_t:
    ZZ_mat[long]  *long
    ZZ_mat[mpz_t] *mpz

ctypedef struct z_nr_t:
    Z_NR[long] long
    Z_NR[mpz_t] mpz

IF HAVE_LONG_DOUBLE:
    IF HAVE_QD:
        # we cannot use a union because of non-trivial constructors
        ctypedef struct fp_nr_t:
            FP_NR[d_t] d
            FP_NR[ld_t] ld
            FP_NR[dpe_t] dpe
            FP_NR[dd_t] dd
            FP_NR[qd_t] qd
            FP_NR[mpfr_t] mpfr
    ELSE:
        ctypedef struct fp_nr_t:
            FP_NR[d_t] d
            FP_NR[ld_t] ld
            FP_NR[dpe_t] dpe
            FP_NR[mpfr_t] mpfr

    IF HAVE_QD:
        ctypedef union mat_gso_core_t:
            MatGSOInterface[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            MatGSOInterface[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            MatGSOInterface[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            MatGSOInterface[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            MatGSOInterface[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            MatGSOInterface[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            MatGSOInterface[Z_NR[long], FP_NR[d_t]] *long_d
            MatGSOInterface[Z_NR[long], FP_NR[ld_t]] *long_ld
            MatGSOInterface[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            MatGSOInterface[Z_NR[long], FP_NR[dd_t]] *long_dd
            MatGSOInterface[Z_NR[long], FP_NR[qd_t]] *long_qd
            MatGSOInterface[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union mat_gso_core_t:
            MatGSOInterface[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            MatGSOInterface[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            MatGSOInterface[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            MatGSOInterface[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            MatGSOInterface[Z_NR[long], FP_NR[d_t]] *long_d
            MatGSOInterface[Z_NR[long], FP_NR[ld_t]] *long_ld
            MatGSOInterface[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            MatGSOInterface[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union lll_reduction_core_t:
            LLLReduction[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            LLLReduction[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            LLLReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            LLLReduction[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            LLLReduction[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            LLLReduction[Z_NR[long], FP_NR[d_t]] *long_d
            LLLReduction[Z_NR[long], FP_NR[ld_t]] *long_ld
            LLLReduction[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            LLLReduction[Z_NR[long], FP_NR[dd_t]] *long_dd
            LLLReduction[Z_NR[long], FP_NR[qd_t]] *long_qd
            LLLReduction[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union lll_reduction_core_t:
            LLLReduction[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            LLLReduction[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            LLLReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            LLLReduction[Z_NR[long], FP_NR[d_t]] *long_d
            LLLReduction[Z_NR[long], FP_NR[ld_t]] *long_ld
            LLLReduction[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            LLLReduction[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union bkz_auto_abort_core_t:
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            BKZAutoAbort[Z_NR[long], FP_NR[d_t]] *long_d
            BKZAutoAbort[Z_NR[long], FP_NR[ld_t]] *long_ld
            BKZAutoAbort[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            BKZAutoAbort[Z_NR[long], FP_NR[dd_t]] *long_dd
            BKZAutoAbort[Z_NR[long], FP_NR[qd_t]] *long_qd
            BKZAutoAbort[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union bkz_auto_abort_core_t:
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            BKZAutoAbort[Z_NR[long], FP_NR[d_t]] *long_d
            BKZAutoAbort[Z_NR[long], FP_NR[ld_t]] *long_ld
            BKZAutoAbort[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            BKZAutoAbort[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union bkz_reduction_core_t:
            BKZReduction[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            BKZReduction[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            BKZReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            BKZReduction[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            BKZReduction[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            BKZReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            BKZReduction[Z_NR[long], FP_NR[d_t]] *long_d
            BKZReduction[Z_NR[long], FP_NR[ld_t]] *long_ld
            BKZReduction[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            BKZReduction[Z_NR[long], FP_NR[dd_t]] *long_dd
            BKZReduction[Z_NR[long], FP_NR[qd_t]] *long_qd
            BKZReduction[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union bkz_reduction_core_t:
            BKZReduction[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            BKZReduction[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            BKZReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            BKZReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            BKZReduction[Z_NR[long], FP_NR[d_t]] *long_d
            BKZReduction[Z_NR[long], FP_NR[ld_t]] *long_ld
            BKZReduction[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            BKZReduction[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union fast_evaluator_core_t:
            FastEvaluator[FP_NR[d_t]] *d
            FastEvaluator[FP_NR[ld_t]] *ld
            FastEvaluator[FP_NR[dpe_t]] *dpe
            FastEvaluator[FP_NR[dd_t]] *dd
            FastEvaluator[FP_NR[qd_t]] *qd
            FastErrorBoundedEvaluator *mpfr
    ELSE:
        ctypedef union fast_evaluator_core_t:
            FastEvaluator[FP_NR[d_t]] *d
            FastEvaluator[FP_NR[ld_t]] *ld
            FastEvaluator[FP_NR[dpe_t]] *dpe
            FastErrorBoundedEvaluator *mpfr

    IF HAVE_QD:
        ctypedef union evaluator_core_t:
            Evaluator[FP_NR[d_t]] *d
            Evaluator[FP_NR[ld_t]] *ld
            Evaluator[FP_NR[dpe_t]] *dpe
            Evaluator[FP_NR[dd_t]] *dd
            Evaluator[FP_NR[qd_t]] *qd
            ErrorBoundedEvaluator *mpfr
    ELSE:
        ctypedef union evaluator_core_t:
            Evaluator[FP_NR[d_t]] *d
            Evaluator[FP_NR[ld_t]] *ld
            Evaluator[FP_NR[dpe_t]] *dpe
            ErrorBoundedEvaluator *mpfr

    IF HAVE_QD:
        ctypedef union enumeration_core_t:
            Enumeration[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            Enumeration[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            Enumeration[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            Enumeration[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            Enumeration[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            Enumeration[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            Enumeration[Z_NR[long], FP_NR[d_t]] *long_d
            Enumeration[Z_NR[long], FP_NR[ld_t]] *long_ld
            Enumeration[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            Enumeration[Z_NR[long], FP_NR[dd_t]] *long_dd
            Enumeration[Z_NR[long], FP_NR[qd_t]] *long_qd
            Enumeration[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union enumeration_core_t:
            Enumeration[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            Enumeration[Z_NR[mpz_t], FP_NR[ld_t]] *mpz_ld
            Enumeration[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            Enumeration[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            Enumeration[Z_NR[long], FP_NR[d_t]] *long_d
            Enumeration[Z_NR[long], FP_NR[ld_t]] *long_ld
            Enumeration[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            Enumeration[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union pruner_core_t:
            Pruner[FP_NR[d_t]] *d
            Pruner[FP_NR[ld_t]] *ld
            Pruner[FP_NR[dpe_t]] *dpe
            Pruner[FP_NR[dd_t]] *dd
            Pruner[FP_NR[qd_t]] *qd
            Pruner[FP_NR[mpfr_t]] *mpfr
    ELSE:
        ctypedef union pruner_core_t:
            Pruner[FP_NR[d_t]] *d
            Pruner[FP_NR[ld_t]] *ld
            Pruner[FP_NR[dpe_t]] *dpe
            Pruner[FP_NR[mpfr_t]] *mpfr

    IF HAVE_QD:
        # we cannot use a union because of non-trivial constructors
        ctypedef struct vector_fp_nr_t:
            vector[FP_NR[d_t]] d
            vector[FP_NR[ld_t]] ld
            vector[FP_NR[dpe_t]] dpe
            vector[FP_NR[dd_t]] dd
            vector[FP_NR[qd_t]] qd
            vector[FP_NR[mpfr_t]] mpfr
    ELSE:
        ctypedef struct vector_fp_nr_t:
            vector[FP_NR[d_t]] d
            vector[FP_NR[ld_t]] ld
            vector[FP_NR[dpe_t]] dpe
            vector[FP_NR[mpfr_t]] mpfr
ELSE:
    IF HAVE_QD:
        # we cannot use a union because of non-trivial constructors
        ctypedef struct fp_nr_t:
            FP_NR[d_t] d
            FP_NR[dpe_t] dpe
            FP_NR[dd_t] dd
            FP_NR[qd_t] qd
            FP_NR[mpfr_t] mpfr
    ELSE:
        ctypedef struct fp_nr_t:
            FP_NR[d_t] d
            FP_NR[dpe_t] dpe
            FP_NR[mpfr_t] mpfr

    IF HAVE_QD:
        ctypedef union mat_gso_core_t:
            MatGSOInterface[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            MatGSOInterface[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            MatGSOInterface[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            MatGSOInterface[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            MatGSOInterface[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            MatGSOInterface[Z_NR[long], FP_NR[d_t]] *long_d
            MatGSOInterface[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            MatGSOInterface[Z_NR[long], FP_NR[dd_t]] *long_dd
            MatGSOInterface[Z_NR[long], FP_NR[qd_t]] *long_qd
            MatGSOInterface[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union mat_gso_core_t:
            MatGSOInterface[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            MatGSOInterface[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            MatGSOInterface[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            MatGSOInterface[Z_NR[long], FP_NR[d_t]] *long_d
            MatGSOInterface[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            MatGSOInterface[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union lll_reduction_core_t:
            LLLReduction[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            LLLReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            LLLReduction[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            LLLReduction[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            LLLReduction[Z_NR[long], FP_NR[d_t]] *long_d
            LLLReduction[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            LLLReduction[Z_NR[long], FP_NR[dd_t]] *long_dd
            LLLReduction[Z_NR[long], FP_NR[qd_t]] *long_qd
            LLLReduction[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union lll_reduction_core_t:
            LLLReduction[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            LLLReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            LLLReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            LLLReduction[Z_NR[long], FP_NR[d_t]] *long_d
            LLLReduction[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            LLLReduction[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union bkz_auto_abort_core_t:
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            BKZAutoAbort[Z_NR[long], FP_NR[d_t]] *long_d
            BKZAutoAbort[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            BKZAutoAbort[Z_NR[long], FP_NR[dd_t]] *long_dd
            BKZAutoAbort[Z_NR[long], FP_NR[qd_t]] *long_qd
            BKZAutoAbort[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union bkz_auto_abort_core_t:
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            BKZAutoAbort[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            BKZAutoAbort[Z_NR[long], FP_NR[d_t]] *long_d
            BKZAutoAbort[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            BKZAutoAbort[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union bkz_reduction_core_t:
            BKZReduction[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            BKZReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            BKZReduction[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            BKZReduction[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            BKZReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            BKZReduction[Z_NR[long], FP_NR[d_t]] *long_d
            BKZReduction[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            BKZReduction[Z_NR[long], FP_NR[dd_t]] *long_dd
            BKZReduction[Z_NR[long], FP_NR[qd_t]] *long_qd
            BKZReduction[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union bkz_reduction_core_t:
            BKZReduction[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            BKZReduction[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            BKZReduction[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            BKZReduction[Z_NR[long], FP_NR[d_t]] *long_d
            BKZReduction[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            BKZReduction[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union fast_evaluator_core_t:
            FastEvaluator[FP_NR[d_t]] *d
            FastEvaluator[FP_NR[dpe_t]] *dpe
            FastEvaluator[FP_NR[dd_t]] *dd
            FastEvaluator[FP_NR[qd_t]] *qd
            FastErrorBoundedEvaluator *mpfr
    ELSE:
        ctypedef union fast_evaluator_core_t:
            FastEvaluator[FP_NR[d_t]] *d
            FastEvaluator[FP_NR[dpe_t]] *dpe
            FastErrorBoundedEvaluator *mpfr

    IF HAVE_QD:
        ctypedef union evaluator_core_t:
            Evaluator[FP_NR[d_t]] *d
            Evaluator[FP_NR[dpe_t]] *dpe
            Evaluator[FP_NR[dd_t]] *dd
            Evaluator[FP_NR[qd_t]] *qd
            ErrorBoundedEvaluator *mpfr
    ELSE:
        ctypedef union evaluator_core_t:
            Evaluator[FP_NR[d_t]] *d
            Evaluator[FP_NR[dpe_t]] *dpe
            ErrorBoundedEvaluator *mpfr

    IF HAVE_QD:
        ctypedef union enumeration_core_t:
            Enumeration[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            Enumeration[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            Enumeration[Z_NR[mpz_t], FP_NR[dd_t]] *mpz_dd
            Enumeration[Z_NR[mpz_t], FP_NR[qd_t]] *mpz_qd
            Enumeration[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            Enumeration[Z_NR[long], FP_NR[d_t]] *long_d
            Enumeration[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            Enumeration[Z_NR[long], FP_NR[dd_t]] *long_dd
            Enumeration[Z_NR[long], FP_NR[qd_t]] *long_qd
            Enumeration[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr
    ELSE:
        ctypedef union enumeration_core_t:
            Enumeration[Z_NR[mpz_t], FP_NR[d_t]] *mpz_d
            Enumeration[Z_NR[mpz_t], FP_NR[dpe_t]] *mpz_dpe
            Enumeration[Z_NR[mpz_t], FP_NR[mpfr_t]] *mpz_mpfr
            Enumeration[Z_NR[long], FP_NR[d_t]] *long_d
            Enumeration[Z_NR[long], FP_NR[dpe_t]] *long_dpe
            Enumeration[Z_NR[long], FP_NR[mpfr_t]] *long_mpfr

    IF HAVE_QD:
        ctypedef union pruner_core_t:
            Pruner[FP_NR[d_t]] *d
            Pruner[FP_NR[dpe_t]] *dpe
            Pruner[FP_NR[dd_t]] *dd
            Pruner[FP_NR[qd_t]] *qd
            Pruner[FP_NR[mpfr_t]] *mpfr
    ELSE:
        ctypedef union pruner_core_t:
            Pruner[FP_NR[d_t]] *d
            Pruner[FP_NR[dpe_t]] *dpe
            Pruner[FP_NR[mpfr_t]] *mpfr

    IF HAVE_QD:
        # we cannot use a union because of non-trivial constructors
        ctypedef struct vector_fp_nr_t:
            vector[FP_NR[d_t]] d
            vector[FP_NR[dpe_t]] dpe
            vector[FP_NR[dd_t]] dd
            vector[FP_NR[qd_t]] qd
            vector[FP_NR[mpfr_t]] mpfr
    ELSE:
        ctypedef struct vector_fp_nr_t:
            vector[FP_NR[d_t]] d
            vector[FP_NR[dpe_t]] dpe
            vector[FP_NR[mpfr_t]] mpfr

ctypedef struct vector_z_nr_t:
    vector[Z_NR[mpz_t]] mpz
    vector[Z_NR[long]] long
