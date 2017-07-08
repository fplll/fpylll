# -*- coding: utf-8 -*-
"""
Pruner

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>

    >>> from fpylll import *
    >>> set_random_seed(1337)
    >>> A = [IntegerMatrix.random(10, "qary", bits=10, k=5) for _ in range(20)]
    >>> M = [GSO.Mat(a) for a in A]
    >>> _ = [LLL.Reduction(m)() for m in M]
    >>> radius = sum([m.get_r(0, 0) for m in M])/len(M)
    >>> pr = prune(radius, 10000, 0.4, [m.r() for m in M])
    >>> print(pr)  # doctest: +ELLIPSIS
    Pruning<1.000000, (1.00,...,0.79), 0.3983>

    >>> print(prune(M[0].get_r(0, 0), 2**20, 0.9, [m.r() for m in M], pruning=pr))
    Pruning<1.000000, (1.00,...,0.89), 0.9302>

"""

include "fpylll/config.pxi"

from libcpp cimport bool
from libcpp.vector cimport vector
from math import log, exp
from cysignals.signals cimport sig_on, sig_off

from decl cimport fp_nr_t, mpz_t, dpe_t, mpfr_t
from decl cimport nr_d, nr_dpe, nr_mpfr, pruner_core_t
from bkz_param cimport Pruning
from fplll cimport FT_DOUBLE, FT_DPE, FT_MPFR, FloatType
from fplll cimport PRUNER_METHOD_GRADIENT, PRUNER_METHOD_NM, PRUNER_METHOD_HYBRID, PRUNER_METHOD_GREEDY
from fplll cimport PRUNER_METRIC_PROBABILITY_OF_SHORTEST, PRUNER_METRIC_EXPECTED_SOLUTIONS
from fplll cimport FP_NR, Z_NR
from fplll cimport MatGSO as MatGSO_c
from fplll cimport prune as prune_c
from fplll cimport Pruning as Pruning_c
from fplll cimport Pruner as Pruner_c
from fplll cimport svp_probability as svp_probability_c
from fpylll.util import adjust_radius_to_gh_bound, precision, get_precision
from fpylll.util cimport check_float_type, check_precision, check_descent_method, check_pruner_metric

IF HAVE_LONG_DOUBLE:
    from fplll cimport FT_LONG_DOUBLE
    from decl cimport gso_mpz_ld
    from decl cimport nr_ld

IF HAVE_QD:
    from fpylll.qd.qd cimport dd_real, qd_real
    from decl cimport gso_mpz_dd, gso_mpz_qd
    from decl cimport nr_dd, nr_qd
    from fplll cimport FT_DD, FT_QD

from bkz_param cimport Pruning
from gso cimport MatGSO

cdef class Pruner:
    def __init__(self, double enumeration_radius, double preproc_cost, double target,
                 method="gradient", metric="probability", size_t n=0, size_t d=0,
                 float_type="double"):
        """

        :param enumeration_radius: target squared enumeration radius
        :param preproc_cost:       cost of preprocessing
        :param target:             overall targeted success probability or number of solutions
        :param method:             one of "gradient", "nm", "greedy" or "hybrid"
        :param metric:             "probability" or "solutions"
        :param n:                  dimension
        :param d:                  dimension/2
        :param float_type:         floating point type to use

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)

        .. note :: Preprocessing cost should be expressed in terms of nodes in an
           enumeration (~100 CPU cycles per node)

        """
        cdef FloatType float_type_ = check_float_type(float_type)
        method = check_descent_method(method)
        metric = check_pruner_metric(metric)
        cdef fp_nr_t enumeration_radius_
        cdef fp_nr_t preproc_cost_
        cdef fp_nr_t target_

        if preproc_cost < 1:
            raise ValueError("Preprocessing cost must be at least 1 but got %f"%preproc_cost)
        if metric == PRUNER_METRIC_PROBABILITY_OF_SHORTEST:
            if target <= 0 or target >= 1.0:
                raise ValueError("Probability must be between 0 and 1 (exclusive) but got %f"%target)
        if metric == PRUNER_METRIC_EXPECTED_SOLUTIONS:
            if target <= 0 or target >= 1.0:
                raise ValueError("Number of solutions must be > 0 but got %f"%target)

        if float_type_ == FT_DOUBLE:
            self._type = nr_d
            enumeration_radius_.d = enumeration_radius
            preproc_cost_.d = preproc_cost
            target_.d = target
            self._core.d = new Pruner_c[FP_NR[double]](enumeration_radius_.d, preproc_cost_.d, target_.d,
                                                       method, metric, n, d)
        elif float_type_ == FT_LONG_DOUBLE:
            IF HAVE_LONG_DOUBLE:
                self._type = nr_ld
                enumeration_radius_.ld = enumeration_radius
                preproc_cost_.ld = preproc_cost
                target_.ld = target
                self._core.ld = new Pruner_c[FP_NR[longdouble]](enumeration_radius_.ld, preproc_cost_.ld, target_.ld,
                                                                method, metric, n, d)
            ELSE:
                raise ValueError("Float type '%s' not understood." % float_type)
        elif float_type_ == FT_DPE:
            self._type = nr_dpe
            enumeration_radius_.dpe = enumeration_radius
            preproc_cost_.dpe = preproc_cost
            target_.dpe = target
            self._core.dpe = new Pruner_c[FP_NR[dpe_t]](enumeration_radius_.dpe, preproc_cost_.dpe, target_.dpe,
                                                        method, metric, n, d)
        elif float_type_ == FT_MPFR:
            self._type = nr_mpfr
            enumeration_radius_.mpfr = enumeration_radius
            preproc_cost_.mpfr = preproc_cost
            target_.mpfr = target
            self._core.mpfr = new Pruner_c[FP_NR[mpfr_t]](enumeration_radius_.mpfr, preproc_cost_.mpfr, target_.mpfr,
                                                          method, metric, n, d)

        else:
            IF HAVE_QD:
                if float_type_ == FT_DD:
                    self._type = nr_dd
                    enumeration_radius_.dd = enumeration_radius
                    preproc_cost_.dd = preproc_cost
                    target_.dd = target
                    self._core.dd = new Pruner_c[FP_NR[dd_real]](enumeration_radius_.dd, preproc_cost_.dd, target_.dd,
                                                                 method, metric, n, d)

                elif float_type_ == FT_QD:
                    self._type = nr_qd
                    enumeration_radius_.qd = enumeration_radius
                    preproc_cost_.qd = preproc_cost
                    target_.qd = target
                    self._core.qd = new Pruner_c[FP_NR[qd_real]](enumeration_radius_.qd, preproc_cost_.qd, target_.qd,
                                                                 method, metric, n, d)
                else:
                    raise ValueError("Float type '%s' not understood."%float_type)
            ELSE:
                raise ValueError("Float type '%s' not understood."%float_type)

    def __dealloc__(self):
        if self._type == nr_d:
            del self._core.d
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                del self._core.ld
        if self._type == nr_dpe:
            del self._core.dpe
        IF HAVE_QD:
            if self._type == nr_dd:
                del self._core.dd
            if self._type == nr_qd:
                del self._core.qd
        if self._type == nr_mpfr:
            del self._core.mpfr

    @property
    def enumeration_radius(self):
        """
        Enumeration radius (squared).

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)
        >>> pr.enumeration_radius
        1912130.0

        """
        if self._type == nr_d:
            return self._core.d.enumeration_radius.get_d()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                return self._core.ld.enumeration_radius.get_d()
        if self._type == nr_dpe:
            return self._core.dpe.enumeration_radius.get_d()
        IF HAVE_QD:
            if self._type == nr_dd:
                return self._core.dd.enumeration_radius.get_d()
            if self._type == nr_qd:
                return self._core.qd.enumeration_radius.get_d()
        if self._type == nr_mpfr:
            return self._core.mpfr.enumeration_radius.get_d()

        raise RuntimeError("Pruner object '%s' has no core."%self)

    @property
    def preproc_cost(self):
        """
        Cost of pre-processing a basis for a retrial.

        This cost is expressed in terms of nodes in an enumeration.

        Roughly, a node is equivalent to 100 CPU cycles.

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)
        >>> pr.preproc_cost
        1048576.0

        """
        if self._type == nr_d:
            return self._core.d.preproc_cost.get_d()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                return self._core.ld.preproc_cost.get_d()
        if self._type == nr_dpe:
            return self._core.dpe.preproc_cost.get_d()
        IF HAVE_QD:
            if self._type == nr_dd:
                return self._core.dd.preproc_cost.get_d()
            if self._type == nr_qd:
                return self._core.qd.preproc_cost.get_d()
        if self._type == nr_mpfr:
            return self._core.mpfr.preproc_cost.get_d()

        raise RuntimeError("Pruner object '%s' has no core."%self)

    @property
    def target(self):
        """
        Overall target success probability/expected solutions

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)
        >>> pr.target
        0.51

        """
        if self._type == nr_d:
            return self._core.d.target.get_d()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                return self._core.ld.target.get_d()
        if self._type == nr_dpe:
            return self._core.dpe.target.get_d()
        IF HAVE_QD:
            if self._type == nr_dd:
                return self._core.dd.target.get_d()
            if self._type == nr_qd:
                return self._core.qd.target.get_d()
        if self._type == nr_mpfr:
            return self._core.mpfr.target.get_d()

        raise RuntimeError("Pruner object '%s' has no core."%self)

    def load_basis_shapes(self, gso_sq_norms):
        """
        Load the shape of a basis from a tuple.

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)
        >>> pr.load_basis_shapes([[M.get_r(i,i) for i in range(M.d)]])

        """
        cdef vector[vector[double]] vec

        d = len(gso_sq_norms[0])
        for i,m in enumerate(gso_sq_norms):
            vec.push_back(vector[double]())
            if len(m) != d:
                raise ValueError("Lengths of all vectors must match.")
            for e in m:
                vec[i].push_back(e)

        if self._type == nr_d:
            return self._core.d.load_basis_shapes(vec)
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                return self._core.ld.load_basis_shapes(vec)
        if self._type == nr_dpe:
            return self._core.dpe.load_basis_shapes(vec)
        IF HAVE_QD:
            if self._type == nr_dd:
                return self._core.dd.load_basis_shapes(vec)
            if self._type == nr_qd:
                return self._core.qd.load_basis_shapes(vec)
        if self._type == nr_mpfr:
            return self._core.mpfr.load_basis_shapes(vec)

        raise RuntimeError("Pruner object '%s' has no core."%self)

    def optimize_coefficients(self, pr, reset=True):
        """
        Optimize pruning coefficients.

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)
        >>> pr.load_basis_shapes([[M.get_r(i,i) for i in range(M.d)]])
        >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
        >>> c[0:10]  # doctest: +ELLIPSIS
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95658..., 0.95658..., 0.95397..., 0.95397...)

        .. note :: Basis shape and other parameters must have been set beforehand.

        """
        cdef vector[double] pr_
        cdef bool called = False

        d = len(pr)
        for e in pr:
            pr_.push_back(e)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            self._core.d.optimize_coefficients(pr_, reset)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                self._core.ld.optimize_coefficients(pr_, reset)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            self._core.dpe.optimize_coefficients(pr_, reset)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                self._core.dd.optimize_coefficients(pr_, reset)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                self._core.qd.optimize_coefficients(pr_, reset)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            self._core.mpfr.optimize_coefficients(pr_,  reset)
            called = True
            sig_off()

        if not called:
             raise RuntimeError("Pruner object '%s' has no core."%self)

        pr = []
        for i in range(d):
            pr.append(pr_[i])
        return tuple(pr)

    def single_enum_cost(self, pr, detailed_cost=False):
        """
        Compute the cost of a single enumeration

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)
        >>> pr.load_basis_shapes([[M.get_r(i,i) for i in range(M.d)]])
        >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
        >>> cost, details = pr.single_enum_cost(c, True)
        >>> cost  # doctest: +ELLIPSIS
        20689.8...

        >>> details[0:10]  # doctest: +ELLIPSIS
        (0.141..., 0.330..., 0.914..., 2.203..., 5.885..., 16.687..., 21.318..., 49.025..., 108.852..., 218.762...)

        """
        cdef vector[double] pr_
        cdef vector[double] detailed_cost_
        cdef bool called = False
        cost = 0.0

        d = len(pr)
        for e in pr:
            pr_.push_back(e)
            detailed_cost_.push_back(0.0)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            cost = self._core.d.single_enum_cost(pr_, &detailed_cost_)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                cost = self._core.ld.single_enum_cost(pr_, &detailed_cost_)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            cost = self._core.dpe.single_enum_cost(pr_, &detailed_cost_)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                cost = self._core.dd.single_enum_cost(pr_, &detailed_cost_)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                cost = self._core.qd.single_enum_cost(pr_, &detailed_cost_)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            cost = self._core.mpfr.single_enum_cost(pr_, &detailed_cost_)
            called = True
            sig_off()

        if not called:
             raise RuntimeError("Pruner object '%s' has no core."%self)

        if detailed_cost:
            detailed_cost = []
            for i in range(d):
                detailed_cost.append(detailed_cost_[i])
            return cost, tuple(detailed_cost)
        else:
            return cost

    def repeated_enum_cost(self, pr):
        """
        Compute the cost of r enumeration and (r-1) preprocessing, where r is the required number of
        retrials to reach target

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)
        >>> pr.load_basis_shapes([[M.get_r(i,i) for i in range(M.d)]])
        >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
        >>> pr.repeated_enum_cost(c)  # doctest: +ELLIPSIS
        20689.89...

        """
        cdef vector[double] pr_
        cdef bool called = False
        cost = 0.0

        for e in pr:
            pr_.push_back(e)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            cost = self._core.d.repeated_enum_cost(pr_)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                cost = self._core.ld.repeated_enum_cost(pr_)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            cost = self._core.dpe.repeated_enum_cost(pr_)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                cost = self._core.dd.repeated_enum_cost(pr_)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                cost = self._core.qd.repeated_enum_cost(pr_)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            cost = self._core.mpfr.repeated_enum_cost(pr_,)
            called = True
            sig_off()

        if not called:
             raise RuntimeError("Pruner object '%s' has no core."%self)

        return cost

    def measure_metric(self, pr):
        """
        Compute the success probability of expected number of solutions of a single enumeration.

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruner, set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pr = Pruner(M.get_r(0,0), 2**20, 0.51)
        >>> pr.load_basis_shapes([[M.get_r(i,i) for i in range(M.d)]])
        >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
        >>> pr.measure_metric(c)  # doctest: +ELLIPSIS
        0.54120...

        """
        cdef vector[double] pr_
        cdef bool called = False
        r = 0.0

        for e in pr:
            pr_.push_back(e)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            r = self._core.d.measure_metric(pr_)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                r = self._core.ld.measure_metric(pr_)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            r = self._core.dpe.measure_metric(pr_)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                r = self._core.dd.measure_metric(pr_)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                r = self._core.qd.measure_metric(pr_)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            r = self._core.mpfr.measure_metric(pr_,)
            called = True
            sig_off()

        if not called:
             raise RuntimeError("Pruner object '%s' has no core."%self)

        return r

def prune(double enumeration_radius, double preproc_cost, double target, M,
          descent_method="gradient", metric="probability", float_type="double", pruning=None):
    """Return optimal pruning parameters.

    :param pruning:            write output here, pass ``None`` for creating a new one
    :param enumeration_radius: target squared enumeration radius
    :param preproc_cost:       cost of preprocessing
    :param target:             overall targeted success probability or number of solutions
    :param M:                  list (of lists) with r coefficients
    :param descent_method:     one of "gradient", "nm", "greedy" or "hybrid"
    :param metric:             "probability" or "solutions"
    :param float_type:         floating point type to use

    >>> from fpylll import IntegerMatrix, LLL, GSO, get_precision, set_precision
    >>> from fpylll.numpy import dump_r
    >>> from fpylll import set_random_seed
    >>> set_random_seed(1337)
    >>> A = IntegerMatrix.random(20, "qary", bits=20, k=10)
    >>> M = GSO.Mat(A)
    >>> LLL.Reduction(M)()
    >>> _ = set_precision(53)
    >>> R = [M.get_r(i,i) for i in range(0, 20)]
    >>> pr0 = prune(R[0], 2**20, 0.5, [R], float_type="double")
    >>> pr1 = prune(R[0], 2**20, 0.5, [R], float_type="long double")

    >>> pr0.coefficients[10], pr1.coefficients[10] # doctest: +ELLIPSIS
    (0.9242997148513628, 0.9242997148513968)

    >>> pr0 = prune(R[0], 2**10, 0.5, [R], descent_method="nm", float_type="double")
    >>> pr1 = prune(R[0], 2**10, 0.5, [R], descent_method="nm", float_type="long double")
    >>> pr0.coefficients[10], pr1.coefficients[10]
    (0.6780854606138969, 0.6780854606138966)

    .. note :: Preprocessing cost should be expressed in terms of nodes in an
       enumeration (~100 CPU cycles per node)

    """
    if preproc_cost < 1:
        raise ValueError("Preprocessing cost must be at least 1 but got %f"%preproc_cost)
    if metric == PRUNER_METRIC_PROBABILITY_OF_SHORTEST:
        if target <= 0 or target >= 1.0:
            raise ValueError("Probability must be between 0 and 1 (exclusive) but got %f"%target)
    if metric == PRUNER_METRIC_EXPECTED_SOLUTIONS:
        if target <= 0 or target >= 1.0:
            raise ValueError("Number of solutions must be > 0 but got %f"%target)

    cdef FloatType ft = check_float_type(float_type)
    descent_method = check_descent_method(descent_method)
    metric = check_pruner_metric(metric)

    try:
        M[0][0]
    except (AttributeError, TypeError):
        M = [M]

    reset = False
    if pruning is None:
        pruning = Pruning(1.0, [], 1.0)
        reset = True
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
        if descent_method == PRUNER_METHOD_GREEDY:
            return enumeration_radius, pruning
        return pruning
    IF HAVE_LONG_DOUBLE:
        if ft == FT_LONG_DOUBLE:
            sig_on()
            prune_c[FP_NR[longdouble]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
            sig_off()
            if descent_method == PRUNER_METHOD_GREEDY:
                return enumeration_radius, pruning
            return pruning
    if ft == FT_DPE:
        sig_on()
        prune_c[FP_NR[dpe_t]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
        sig_off()
        if descent_method == PRUNER_METHOD_GREEDY:
            return enumeration_radius, pruning
        return pruning
    if ft == FT_MPFR:
        sig_on()
        prune_c[FP_NR[mpfr_t]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
        sig_off()
        if descent_method == PRUNER_METHOD_GREEDY:
            return enumeration_radius, pruning
        return pruning
    IF HAVE_QD:
            if ft == FT_DD:
                sig_on()
                prune_c[FP_NR[dd_real]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
                sig_off()
                if descent_method == PRUNER_METHOD_GREEDY:
                    return enumeration_radius, pruning
                return pruning
            if ft == FT_QD:
                sig_on()
                prune_c[FP_NR[qd_real]]((<Pruning>pruning)._core, enumeration_radius, preproc_cost, target, vec, descent_method, metric, reset)
                sig_off()
                if descent_method == PRUNER_METHOD_GREEDY:
                    return enumeration_radius, pruning
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
