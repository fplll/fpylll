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
    >>> print(prune(radius, 0, 0.9, M))
    Pruning<1.205907, (1.00,...,0.44), 0.8998>

    >>> print(prune(M[0].get_r(0,0), 0, 0.9, M[0]))
    Pruning<1.205907, (1.00,...,0.46), 0.9002>

"""
from libcpp.vector cimport vector
from math import log, exp

from decl cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr, fp_nr_t, mpz_t, dpe_t, mpfr_t
from bkz_param cimport Pruning
from fplll cimport FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR, FloatType
from fplll cimport FP_NR, Z_NR
from fplll cimport MatGSO as MatGSO_c
from fplll cimport prune as prune_c
from fplll cimport Pruning as Pruning_c
from fplll cimport Pruner
from fplll cimport svp_probability as svp_probability_c
from fpylll.util import gaussian_heuristic, precision, get_precision
from fpylll.util cimport check_float_type, check_precision, check_descent_method, check_pruner_metric

IF HAVE_QD:
    from fpylll.qd.qd cimport dd_real, qd_real
    from decl cimport mpz_dd, mpz_qd
    from fplll cimport FT_DD, FT_QD

from bkz_param cimport Pruning
from gso cimport MatGSO

def prune(pruning, double enumeration_radius, double preproc_cost, double target, M,
          descent_method="gradient", metric="probability", float_type="double", reset=True):
    """Return optimal pruning parameters.

    :param pruning:            write output here, pass ``None`` for creating a new one
    :param enumeration_radius: target squared enumeration radius
    :param preproc_cost:       cost of preprocessing
    :param target:             overall targeted success probability or number of solutions
    :param M:                  list (of lists) with r coefficients
    :param descent_method:     one of "gradient", "nm", "greedy" or "hybrid"
    :param metric:             "probability" or "solutions"
    :param float_type:         floating point type to use
    :param reset:              start from current values in ``pruning`` or not

    >>> from fpylll import IntegerMatrix, LLL, GSO, get_precision, set_precision
    >>> from fpylll.numpy import dump_r
    >>> A = IntegerMatrix.random(20, "qary", bits=20, k=10)
    >>> M = GSO.Mat(A)
    >>> LLL.Reduction(M)()
    >>> _ = set_precision(53)
    >>> R = [M.get_r(i,i) for i in range(0, 20)]
    >>> pr0 = prune(R[0], 2^20, 0.5, [R], float_type="double")
    >>> pr1 = prune(R[0], 2^20, 0.5, [R], float_typp="longdouble")

    >>> pr0.coefficients[10], pr1.coefficients[10]
    (0.6074884495481888, 0.6074884494762931)

    >>> pr0 = prune(R[0], 2^20, 0.5, [R], descent_method="nm", float_type="double")
    >>> pr1 = prune(R[0], 2^20, 0.5, [R], descent_method="nm", float_type="longdouble")
    >>> pr0.coefficients[10], pr1.coefficients[10]
    (0.5864317103477773, 0.5864317103476406)
    """

    cdef FloatType ft = check_float_type(float_type)
    descent_method = check_descent_method(descent_method)
    metric = check_pruner_metric(metric)

    try:
        M[0][0]
    except AttributeError:
        M = [M]

    if pruning is None:
        pruning = Pruning(1.0, [], 1.0)
        if reset is False:
            raise ValueError("reset == True requires first parameter != None")
    elif not isinstance(pruning, Pruning):
        raise TypeError("First parameter must be of type Pruning or None but got type '%s'"%type(pruning))


    cdef vector[vector[double]] vec

    d = len(M[0])
    for i,m in enumerate(M):
        vec.push_back(vector[double]())
        if len(m) != d:
            raise ValueError("Lengths of all vectors must match.")
        for e in m:
            vec[i].push_back(e)

    if ft == FT_DOUBLE:
        sig_on()
        prune_c[FP_NR[double]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
        sig_off()
        return pruning
    if ft == FT_LONG_DOUBLE:
        sig_on()
        prune_c[FP_NR[longdouble]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
        sig_off()
        return pruning
    if ft == FT_DPE:
        sig_on()
        prune_c[FP_NR[dpe_t]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
        sig_off()
        return pruning
    if ft == FT_MPFR:
        sig_on()
        prune_c[FP_NR[mpfr_t]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
        sig_off()
        return pruning
    IF HAVE_QD:
            if ft == FT_DD:
                sig_on()
                prune_c[FP_NR[dd_real]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
                sig_off()
                return pruning
            if ft == FT_QD:
                sig_on()
                prune_c[FP_NR[qd_real]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
                sig_off()
                return pruning


def svp_probability(pr, float_type="double"):
    """Return probability of success for enumeration with given set of pruning parameters.

    :param pr: pruning parameters, either Pruning object or list of floating point numbers
    :param float_type: floating point type used internally

    """
    cdef FloatType ft = check_float_type(float_type)

    if not isinstance(pr, Pruning):
        pr = Pruning(1.0, pr, 1.0)

    if ft == FT_DOUBLE:
        return svp_probability_c[FP_NR[double]]((<Pruning>pr)._core.coefficients).get_d()
    IF HAVE_LONG_DOUBLE:
        if ft == FT_LONG_DOUBLE:
            return svp_probability_c[FP_NR[longdouble]]((<Pruning>pr)._core.coefficients).get_d()
    if ft == FT_DPE:
        return svp_probability_c[FP_NR[dpe_t]]((<Pruning>pr)._core.coefficients).get_d()
    if ft == FT_MPFR:
        return svp_probability_c[FP_NR[mpfr_t]]((<Pruning>pr)._core.coefficients).get_d()
    IF HAVE_QD:
        if ft == FT_DD:
            return svp_probability_c[FP_NR[dd_real]]((<Pruning>pr)._core.coefficients).get_d()
        if ft == FT_QD:
            return svp_probability_c[FP_NR[qd_real]]((<Pruning>pr)._core.coefficients).get_d()

    raise ValueError("Float type '%s' not understood."%float_type)
