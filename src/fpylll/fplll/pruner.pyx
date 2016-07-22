# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"

"""
Pruner

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>

    >>> from fpylll import *
    >>> A = [IntegerMatrix.random(10, "qary", bits=10, k=10) for _ in range(20)]
    >>> M = [GSO.Mat(a) for a in A]
    >>> _ = [LLL.Reduction(m)() for m in M]
    >>> radius = sum([m.get_r(0, 0) for m in M])/len(M)
    >>> print prune(radius, 0, 0.9, M)
    Pruning<1.205907, (1.00,...,0.44), 0.8998>

    >>> print prune(M[0].get_r(0,0), 0, 0.9, M[0])
    Pruning<1.205907, (1.00,...,0.46), 0.9002>

"""
from libcpp.vector cimport vector
from math import log, exp

from decl cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr, fp_nr_t, mpz_t, dpe_t, mpfr_t
from fplll cimport FP_NR, Z_NR
from fplll cimport MatGSO as MatGSO_c
from fplll cimport prune as prune_c
from fplll cimport Pruning as Pruning_c
from fplll cimport Pruner
from fpylll.util import gaussian_heuristic


IF HAVE_QD:
    from qd.qd cimport dd_real, qd_real
    from fpylll cimport mpz_dd, mpz_qd

from bkz_param cimport Pruning
from gso cimport MatGSO

def _prune_vec(double enumeration_radius, double preproc_cost, double target_probability, M,
               int start_row = 0, int stop_row=0):
    """Return optimal pruning parameters.

    :param enumeration_radius: target squared enumeration radius
    :param preproc_cost:       cost of preprocessing
    :param target_probability: overall targeted success probability
    :param M:                  GSO object or list of GSO objects
    :param int start_row:      start enumeration in this row
    :param int stop_row:       stop enumeration at this row

    """

    try:
        M[0][0]
    except AttributeError:
        M = [M]

    cdef vector[vector[double]] vec

    n = len(M)
    d = len(M[0])

    avg = [0.0 for _ in range(d)]

    for i,m in enumerate(M):
        vec.push_back(vector[double]())
        if len(m) != d:
            raise ValueError("Lengths of all vectors must match.")
        for j,e in enumerate(m):
            avg[j] += e
            vec[i].push_back(e)

    avg = [e/n for e in avg]

    cdef Pruner[FP_NR[double]] pruner = Pruner[FP_NR[double]](enumeration_radius, preproc_cost, target_probability);
    pruner.load_basis_shapes(vec);

    cdef Pruning_c pruning;

    root_det = exp(sum([log(e) for e in avg])/d)
    gh, expo = gaussian_heuristic(enumeration_radius, 0, d, root_det, 1.0);

    sig_on()
    pruner.optimize_coefficients(pruning.coefficients, True)
    sig_off()
    pruning.probability = pruner.svp_probability(pruning.coefficients)
    pruning.radius_factor = enumeration_radius/(gh * 2**expo)
    return Pruning.from_cxx(pruning)


def _prune_gso(double enumeration_radius, double preproc_cost, double target_probability, M,
               int start_row = 0, int stop_row=0):
    """Return optimal pruning parameters.

    :param enumeration_radius: target squared enumeration radius
    :param preproc_cost:       cost of preprocessing
    :param target_probability: overall targeted success probability
    :param M:                  GSO object or list of GSO objects
    :param int start_row:      start enumeration in this row
    :param int stop_row:       stop enumeration at this row

    """

    if isinstance(M, MatGSO):
        M = [M]
    else:
        M = list(M)

    cdef int type = 0
    cdef Pruning_c pruning_c

    for m in M:
        if not isinstance(m, MatGSO):
            raise TypeError("Parameters must be list of GSO objects.")
        if type == 0:
            type = (<MatGSO>m)._type
        elif (<MatGSO>m)._type != type:
            raise ValueError("Inconsistent cores in parameter list.")

    cdef vector[MatGSO_c[Z_NR[mpz_t], FP_NR[double]]] v_double
    cdef vector[MatGSO_c[Z_NR[mpz_t], FP_NR[longdouble]]] v_ld
    cdef vector[MatGSO_c[Z_NR[mpz_t], FP_NR[dpe_t]]] v_dpe
    IF HAVE_QD:
        cdef vector[MatGSO_c[Z_NR[mpz_t], FP_NR[dd_real]]] v_dd
        cdef vector[MatGSO_c[Z_NR[mpz_t], FP_NR[qd_real]]] v_qd
    cdef vector[MatGSO_c[Z_NR[mpz_t], FP_NR[mpfr_t]]] v_mpfr

    if type == mpz_double:
        for m in M:
            v_double.push_back((<MatGSO>m)._core.mpz_double[0])

        sig_on()
        pruning_c = prune_c[FP_NR[double], Z_NR[mpz_t], FP_NR[double]](enumeration_radius, preproc_cost, target_probability, v_double, start_row, stop_row)
        sig_off()

    elif type == mpz_ld:
        for m in M:
            v_ld.push_back((<MatGSO>m)._core.mpz_ld[0])

        sig_on()
        pruning_c = prune_c[FP_NR[longdouble], Z_NR[mpz_t], FP_NR[longdouble]](enumeration_radius, preproc_cost, target_probability, v_ld, start_row, stop_row)
        sig_off()

    elif type == mpz_dpe:
        for m in M:
            v_dpe.push_back((<MatGSO>m)._core.mpz_dpe[0])

        sig_on()
        pruning_c = prune_c[FP_NR[dpe_t], Z_NR[mpz_t], FP_NR[dpe_t]](enumeration_radius, preproc_cost, target_probability, v_dpe, start_row, stop_row)
        sig_off()

    elif type == mpz_mpfr:
        for m in M:
            v_mpfr.push_back((<MatGSO>m)._core.mpz_mpfr[0])

            sig_on()
            pruning_c = prune_c[FP_NR[mpfr_t], Z_NR[mpz_t], FP_NR[mpfr_t]](enumeration_radius, preproc_cost, target_probability, v_mpfr, start_row, stop_row)
            sig_off()

    else:
        IF HAVE_QD:
            if type == mpz_dd:
                for m in M:
                    v_dd.push_back((<MatGSO>m)._core.mpz_dd[0])

                    sig_on()
                    pruning_c = prune_c[FP_NR[dd_real], Z_NR[mpz_t], FP_NR[dd_real]](enumeration_radius, preproc_cost, target_probability, v_dd, start_row, stop_row)
                    sig_off()

            if type == mpz_qd:
                for m in M:
                    v_qd.push_back((<MatGSO>m)._core.mpz_qd[0])

                    sig_on()
                    pruning_c = prune_c[FP_NR[qd_real], Z_NR[mpz_t], FP_NR[qd_real]](enumeration_radius, preproc_cost, target_probability, v_qd, start_row, stop_row)
                    sig_off()
            else:
                RuntimeError("Unknown type %d."%type)
        ELSE:
            RuntimeError("Unknown type %d."%type)

    return Pruning.from_cxx(pruning_c)

def prune(double enumeration_radius, double preproc_cost, double target_probability, M,
          int start_row = 0, int stop_row=0):
    """Return optimal pruning parameters.

    :param enumeration_radius: target squared enumeration radius
    :param preproc_cost:       cost of preprocessing
    :param target_probability: overall targeted success probability
    :param M:                  GSO object, list of GSO objects or list of lists with r coefficients
    :param int start_row:      start enumeration in this row
    :param int stop_row:       stop enumeration at this row

    """
    try:
        return _prune_gso(enumeration_radius, preproc_cost, target_probability, M, start_row, stop_row)
    except TypeError:
        return _prune_vec(enumeration_radius, preproc_cost, target_probability, M, start_row, stop_row)
