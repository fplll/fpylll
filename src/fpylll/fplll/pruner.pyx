# -*- coding: utf-8 -*-
"""
Pruner

EXAMPLE::

    >>> from fpylll import *
    >>> FPLLL.set_random_seed(1337)
    >>> A = [IntegerMatrix.random(10, "qary", bits=10, k=5) for _ in range(20)]
    >>> M = [GSO.Mat(a) for a in A]
    >>> _ = [LLL.Reduction(m)() for m in M]
    >>> radius = sum([m.get_r(0, 0) for m in M])/len(M)
    >>> pr = Pruning.run(radius, 10000, [m.r() for m in M], 0.4)
    >>> print(pr)  # doctest: +ELLIPSIS
    PruningParams<1.397930, (1.00,...,0.43), 0.4055>


    >>> print(Pruning.run(M[0].get_r(0, 0), 2**20, [m.r() for m in M], 0.9, pruning=pr))
    PruningParams<1.437235, (1.00,...,0.98), 0.9410>



..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "fpylll/config.pxi"

from libcpp cimport bool
from libcpp.vector cimport vector
from math import log, exp
from cysignals.signals cimport sig_on, sig_off
from cython.operator cimport dereference as deref, preincrement as inc

from .decl cimport fp_nr_t, mpz_t, dpe_t, mpfr_t
from .decl cimport nr_d, nr_dpe, nr_mpfr, pruner_core_t, d_t
from .fplll cimport FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR, FloatType
from .fplll cimport PRUNER_METRIC_PROBABILITY_OF_SHORTEST, PRUNER_METRIC_EXPECTED_SOLUTIONS
from .fplll cimport FP_NR, Z_NR
from .fplll cimport prune as prune_c
from .fplll cimport PruningParams as PruningParams_c
from .fplll cimport Pruner as Pruner_c
from .fplll cimport PrunerMetric
from .fplll cimport svp_probability as svp_probability_c
from .fplll cimport PRUNER_CVP, PRUNER_START_FROM_INPUT, PRUNER_GRADIENT, PRUNER_NELDER_MEAD, PRUNER_VERBOSE
from .fplll cimport PRUNER_SINGLE, PRUNER_HALF


from fpylll.util import adjust_radius_to_gh_bound, precision, FPLLL
from fpylll.util cimport check_float_type, check_precision, check_pruner_metric

IF HAVE_LONG_DOUBLE:
    from .decl cimport nr_ld, ld_t

IF HAVE_QD:
    from .decl cimport nr_dd, nr_qd, dd_t, qd_t
    from .fplll cimport FT_DD, FT_QD


cdef class PruningParams:
    """
    Pruning parameters.
    """
    def __init__(self, gh_factor, coefficients, expectation=1.0,
                 metric="probability", detailed_cost=tuple()):
        """Create new pruning parameters object.

        :param gh_factor: ratio of radius to Gaussian heuristic
        :param coefficients:  a list of pruning coefficients
        :param expectation:   success probability or number of solutions
        :param metric:        either "probability" or "solutions"

        """
        if gh_factor <= 0:
            raise ValueError("Radius factor must be > 0")

        cdef PrunerMetric met = <PrunerMetric>check_pruner_metric(metric)

        if met == PRUNER_METRIC_PROBABILITY_OF_SHORTEST:
            if expectation <= 0 or expectation > 1:
                raise ValueError("Probability must be between 0 and 1")

        self._core.gh_factor = gh_factor
        self._core.expectation = expectation
        self._core.metric = met

        for c in coefficients:
            self._core.coefficients.push_back(c)

        for c in detailed_cost:
            self._core.detailed_cost.push_back(c)

    @staticmethod
    cdef PruningParams from_cxx(PruningParams_c& p):
        """
        Load PruningParams object from C++ PruningParams object.

        .. note::

           All data is copied, i.e. `p` can be safely deleted after this function returned.
        """

        cdef PruningParams self = PruningParams(1.0, ())
        self._core = p
        return self

    @staticmethod
    cdef to_cxx(PruningParams_c& self, PruningParams p):
        """
        Store pruning object in C++ pruning object.

        .. note::

           All data is copied, i.e. `p` can be safely deleted after this function returned.
        """
        self.gh_factor = p._core.gh_factor
        self.expectation = p._core.expectation
        self.metric = p._core.metric
        for c in p._core.coefficients:
            self.coefficients.push_back(c)
        for c in p._core.detailed_cost:
            self.detailed_cost.push_back(c)

    @staticmethod
    def LinearPruningParams(block_size, level):
        """
        Set all pruning coefficients to 1, except the last <level>
        coefficients, these will be linearly with slope `-1 / block_size`.

        :param block_size: block size
        :param level: level
        """
        sig_on()
        cdef PruningParams_c p = PruningParams_c.LinearPruningParams(block_size, level)
        sig_off()
        return PruningParams.from_cxx(p)

    def __reduce__(self):
        """
        ::

            >>> from fpylll.fplll.pruner import PruningParams
            >>> import pickle
            >>> print(pickle.loads(pickle.dumps(PruningParams(1.0, [1.0, 0.6, 0.3], 1.0))))
            PruningParams<1.000000, (1.00,...,0.30), 1.0000>

        """
        return PruningParams, (self.gh_factor, self.coefficients, self.expectation, self.metric, self.detailed_cost)

    def __str__(self):
        return "PruningParams<%f, (%.2f,...,%.2f), %.4f>"%(self.gh_factor, self.coefficients[0], self.coefficients[-1], self.expectation)

    @property
    def gh_factor(self):
        """
        ::

            >>> from fpylll.fplll.pruner import PruningParams
            >>> pr = PruningParams(1.0, [1.0, 0.6, 0.3], 0.9)
            >>> pr.gh_factor
            1.0

        """
        return self._core.gh_factor

    @property
    def expectation(self):
        """
        ::

            >>> from fpylll.fplll.pruner import PruningParams
            >>> pr = PruningParams(1.0, [1.0, 0.6, 0.3], 0.9)
            >>> pr.expectation
            0.9

        """
        return self._core.expectation

    @property
    def metric(self):
        """
        ::

            >>> from fpylll.fplll.pruner import PruningParams
            >>> pr = PruningParams(1.0, [1.0, 0.6, 0.3], 0.9)
            >>> pr.metric
            'probability'

        """
        if self._core.metric == PRUNER_METRIC_PROBABILITY_OF_SHORTEST:
            return "probability"
        elif self._core.metric == PRUNER_METRIC_EXPECTED_SOLUTIONS:
            return "solutions"
        else:
            raise NotImplementedError("Metric %d not understood"%self._core.metric)

    @property
    def coefficients(self):
        """
        ::

            >>> from fpylll.fplll.pruner import PruningParams
            >>> pr = PruningParams(1.0, [1.0, 0.6, 0.3], 0.9)
            >>> pr.coefficients
            (1.0, 0.6, 0.3)

        """
        cdef list coefficients = []
        cdef vector[double].iterator it = self._core.coefficients.begin()
        while it != self._core.coefficients.end():
            coefficients.append(deref(it))
            inc(it)
        return tuple(coefficients)

    @property
    def detailed_cost(self):
        """
        ::

            >>> from fpylll.fplll.pruner import PruningParams
            >>> pr = PruningParams(1.0, [1.0, 0.6, 0.3], 0.9)
            >>> pr.detailed_cost
            ()

        """
        cdef list detailed_cost = []
        cdef vector[double].iterator it = self._core.detailed_cost.begin()
        while it != self._core.detailed_cost.end():
            detailed_cost.append(deref(it))
            inc(it)
        return tuple(detailed_cost)

cdef class Pruner:
    def __init__(self, double enumeration_radius, double preproc_cost,
                 gso_r, double target,
                 metric="probability", int flags=PRUNER_GRADIENT,
                 float_type="double"):
        """
        :param enumeration_radius: target squared enumeration radius
        :param preproc_cost: cost of preprocessing
        :param gso_r: r vector of GSO
        :param target: overall targeted success probability or number of solutions
        :param metric: "probability" or "solutions"
        :param flags: flags
        :param float_type: floating point type to use

        EXAMPLE::

            >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
            >>> _ = LLL.reduction(A)
            >>> M = GSO.Mat(A)
            >>> _ = M.update_gso()
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 0.51)
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 1, metric=Pruning.EXPECTED_SOLUTIONS)

        ..  note :: Preprocessing cost should be expressed in terms of nodes in an enumeration (~100
        CPU cycles per node)

        """
        cdef FloatType float_type_ = check_float_type(float_type)
        cdef PrunerMetric metric_ = check_pruner_metric(metric)
        cdef fp_nr_t enumeration_radius_
        cdef fp_nr_t preproc_cost_
        cdef fp_nr_t target_

        if preproc_cost < 1:
            raise ValueError("Preprocessing cost must be at least 1 but got %f"%preproc_cost)
        if metric == PRUNER_METRIC_PROBABILITY_OF_SHORTEST:
            if target <= 0 or target >= 1.0:
                raise ValueError("Probability must be between 0 and 1 (exclusive) but got %f"%target)
        if metric == PRUNER_METRIC_EXPECTED_SOLUTIONS:
            if target <= 0:
                raise ValueError("Number of solutions must be > 0 but got %f"%target)


        cdef vector[vector[double]] gso_r_

        d = len(gso_r[0])
        for i,m in enumerate(gso_r):
            gso_r_.push_back(vector[double]())
            if len(m) != d:
                raise ValueError("Lengths of all vectors must match.")
            for e in m:
                gso_r_[i].push_back(e)

        if float_type_ == FT_DOUBLE:
            self._type = nr_d
            enumeration_radius_.d = enumeration_radius
            preproc_cost_.d = preproc_cost
            target_.d = target
            self._core.d = new Pruner_c[FP_NR[d_t]](enumeration_radius_.d, preproc_cost_.d, gso_r_,
                                                    target_.d, metric_, flags)
        elif float_type_ == FT_LONG_DOUBLE:
            IF HAVE_LONG_DOUBLE:
                self._type = nr_ld
                enumeration_radius_.ld = enumeration_radius
                preproc_cost_.ld = preproc_cost
                target_.ld = target
                self._core.ld = new Pruner_c[FP_NR[ld_t]](enumeration_radius_.ld, preproc_cost_.ld, gso_r_,
                                                          target_.ld, metric_, flags)
            ELSE:
                raise ValueError("Float type '%s' not understood." % float_type)
        elif float_type_ == FT_DPE:
            self._type = nr_dpe
            enumeration_radius_.dpe = enumeration_radius
            preproc_cost_.dpe = preproc_cost
            target_.dpe = target
            self._core.dpe = new Pruner_c[FP_NR[dpe_t]](enumeration_radius_.dpe, preproc_cost_.dpe, gso_r_,
                                                        target_.dpe, metric_, flags)
        elif float_type_ == FT_MPFR:
            self._type = nr_mpfr
            enumeration_radius_.mpfr = enumeration_radius
            preproc_cost_.mpfr = preproc_cost
            target_.mpfr = target
            self._core.mpfr = new Pruner_c[FP_NR[mpfr_t]](enumeration_radius_.mpfr, preproc_cost_.mpfr, gso_r_,
                                                          target_.mpfr, metric_, flags)
        else:
            IF HAVE_QD:
                if float_type_ == FT_DD:
                    self._type = nr_dd
                    enumeration_radius_.dd = enumeration_radius
                    preproc_cost_.dd = preproc_cost
                    target_.dd = target
                    self._core.dd = new Pruner_c[FP_NR[dd_t]](enumeration_radius_.dd, preproc_cost_.dd, gso_r_,
                                                              target_.dd, metric_, flags)

                elif float_type_ == FT_QD:
                    self._type = nr_qd
                    enumeration_radius_.qd = enumeration_radius
                    preproc_cost_.qd = preproc_cost
                    target_.qd = target
                    self._core.qd = new Pruner_c[FP_NR[qd_t]](enumeration_radius_.qd, preproc_cost_.qd, gso_r_,
                                                              target_.qd, metric_, flags)
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

    def optimize_coefficients(self, pr):
        """
        Optimize pruning coefficients.

        >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
        >>> FPLLL.set_random_seed(1337)
        >>> A = IntegerMatrix.random(60, "qary", bits=20, k=30)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()

        >>> pr = Pruning.Pruner(0.9*M.get_r(0,0), 2**40, [M.r()], 0.51, metric=Pruning.PROBABILITY_OF_SHORTEST)
        >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
        >>> pr.measure_metric(c) # doctest: +ELLIPSIS
        0.002711...

        >>> pr = Pruning.Pruner(0.9*M.get_r(0,0), 2**2, [M.r()], 1.0, metric=Pruning.EXPECTED_SOLUTIONS)
        >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
        >>> pr.measure_metric(c) # doctest: +ELLIPSIS
        0.990517...

        >>> pr = Pruning.Pruner(0.5*M.get_r(0,0), 2**40, [M.r()], 0.51, metric=Pruning.PROBABILITY_OF_SHORTEST, flags=Pruning.SINGLE)
        >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
        >>> pr.measure_metric(c) # doctest: +ELLIPSIS
        0.515304...

        >>> pr = Pruning.Pruner(0.9*M.get_r(0,0), 2**2, [M.r()], 1.0, metric=Pruning.EXPECTED_SOLUTIONS, flags=Pruning.SINGLE)
        >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
        >>> pr.measure_metric(c) # doctest: +ELLIPSIS
        1.043578...

        """
        cdef vector[double] pr_
        cdef bool called = False

        d = len(pr)
        for e in pr:
            pr_.push_back(e)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            self._core.d.optimize_coefficients(pr_)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                self._core.ld.optimize_coefficients(pr_)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            self._core.dpe.optimize_coefficients(pr_)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                self._core.dd.optimize_coefficients(pr_)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                self._core.qd.optimize_coefficients(pr_)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            self._core.mpfr.optimize_coefficients(pr_)
            called = True
            sig_off()

        if not called:
             raise RuntimeError("Pruner object '%s' has no core."%self)

        pr = []
        for i in range(d):
            pr.append(pr_[i])
        return tuple(pr)

    def optimize_coefficients_evec(self, pr):
        """
        Optimize using "even" coefficients.

        Run the optimization process, successively using the algorithm activated using using half
        coefficients: the input ``pr`` has length ``n``; but only the even indices in the vector
        will be used in the optimization.  In the end, we have ``pr_i = pr_{i+1}``.

        This function only optimizes the overall enumeration time where the target function is:

        ``single_enum_cost(pr) * trials + preproc_cost * (trials - 1.0)``

        :param pr: input pruning parameters

        EXAMPLE::

            >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
            >>> _ = LLL.reduction(A)
            >>> M = GSO.Mat(A)
            >>> _ = M.update_gso()
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 0.51)
            >>> c = pr.optimize_coefficients_evec([1.  for _ in range(M.d)])
            >>> c[0:10] # doctest: +ELLIPSIS
            (1.0, 1.0, 0.98, 0.98, 0.98, 0.98, 0.9637..., 0.9637..., 0.9591..., 0.9591...)

        """
        cdef vector[double] pr_
        cdef bool called = False

        d = len(pr)
        for e in pr:
            pr_.push_back(e)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            self._core.d.optimize_coefficients_evec(pr_)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                self._core.ld.optimize_coefficients_evec(pr_)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            self._core.dpe.optimize_coefficients_evec(pr_)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                self._core.dd.optimize_coefficients_evec(pr_)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                self._core.qd.optimize_coefficients_evec(pr_)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            self._core.mpfr.optimize_coefficients_evec(pr_)
            called = True
            sig_off()

        if not called:
             raise RuntimeError("Pruner object '%s' has no core."%self)

        pr = []
        for i in range(d):
            pr.append(pr_[i])
        return tuple(pr)

    def optimize_coefficients_full(self, pr):
        """
        Optimize pruning coefficients using all the coefficients.

        Run the optimization process, successively using the algorithm activated using using full
        coefficients.  That is, we do not have the constraint pr_i = pr_{i+1} in this function.

        Note that this function (and `optimize_coefficients_full_core()`) only optimizes the overall
        enumeration time where the target function is:

        ``single_enum_cost(pr) * trials + preproc_cost * (trials - 1.0)``

        :param pr: input pruning parameters

        EXAMPLE::

            >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
            >>> _ = LLL.reduction(A)
            >>> M = GSO.Mat(A)
            >>> _ = M.update_gso()
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 0.51)
            >>> c = pr.optimize_coefficients_full([1. for _ in range(M.d)])
            >>> c[0:10]  # doctest: +ELLIPSIS
            (1.0, 1.0, 0.98, 0.98, 0.98, 0.98, 0.9608..., 0.9607..., 0.9574..., 0.9572...)

            ..  note :: Basis shape and other parameters must have been set beforehand.
        """
        cdef vector[double] pr_
        cdef bool called = False

        d = len(pr)
        for e in pr:
            pr_.push_back(e)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            self._core.d.optimize_coefficients_full(pr_)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                self._core.ld.optimize_coefficients_full(pr_)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            self._core.dpe.optimize_coefficients_full(pr_)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                self._core.dd.optimize_coefficients_full(pr_)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                self._core.qd.optimize_coefficients_full(pr_)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            self._core.mpfr.optimize_coefficients_full(pr_)
            called = True
            sig_off()

        if not called:
             raise RuntimeError("Pruner object '%s' has no core."%self)

        pr = []
        for i in range(d):
            pr.append(pr_[i])
        return tuple(pr)

    def optimize_coefficients_cost_vary_prob(self, pr):
        """
        Optimize the pruning coefficients with respect to the overall enumeration time.

        The target function is: ``single_enum_cost(pr) * trials + preproc_cost * (trials - 1.0)``;

        EXAMPLE::

            >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
            >>> _ = LLL.reduction(A)
            >>> M = GSO.Mat(A)
            >>> _ = M.update_gso()
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 0.51)
            >>> c = pr.optimize_coefficients_cost_vary_prob([1. for _ in range(M.d)])
            >>> c[0:10]  # doctest: +ELLIPSIS
            (1.0, 1.0, 0.999..., 0.999..., 0.995..., 0.993..., 0.977..., 0.962..., 0.936..., 0.913...)

        """
        cdef vector[double] pr_
        cdef bool called = False

        d = len(pr)
        for e in pr:
            pr_.push_back(e)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            self._core.d.optimize_coefficients_cost_vary_prob(pr_)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                self._core.ld.optimize_coefficients_cost_vary_prob(pr_)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            self._core.dpe.optimize_coefficients_cost_vary_prob(pr_)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                self._core.dd.optimize_coefficients_cost_vary_prob(pr_)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                self._core.qd.optimize_coefficients_cost_vary_prob(pr_)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            self._core.mpfr.optimize_coefficients_cost_vary_prob(pr_)
            called = True
            sig_off()

        if not called:
             raise RuntimeError("Pruner object '%s' has no core."%self)

        pr = []
        for i in range(d):
            pr.append(pr_[i])
        return tuple(pr)


    def optimize_coefficients_cost_fixed_prob(self, pr):
        """
        Optimize pruning coefficients with respect to the single enumeration.

        Main interface to optimize the single enumeration time with the constraint such that the succ.
        prob (or expected solutions) is fixed (and given) from input to the Pruner constructor.

        EXAMPLE::

            >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
            >>> _ = LLL.reduction(A)
            >>> M = GSO.Mat(A)
            >>> _ = M.update_gso()
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 0.51)
            >>> c = pr.optimize_coefficients_cost_fixed_prob([1. for _ in range(M.d)])
            >>> c[0:10]  # doctest: +ELLIPSIS
            (1.0, 1.0, 0.98, 0.98, 0.98, 0.98, 0.962..., 0.944..., 0.944..., 0.944...)

        """
        cdef vector[double] pr_
        cdef bool called = False

        d = len(pr)
        for e in pr:
            pr_.push_back(e)

        # TODO: don't just return doubles
        if self._type == nr_d:
            sig_on()
            self._core.d.optimize_coefficients_cost_fixed_prob(pr_)
            called = True
            sig_off()
        IF HAVE_LONG_DOUBLE:
            if self._type == nr_ld:
                sig_on()
                self._core.ld.optimize_coefficients_cost_fixed_prob(pr_)
                called = True
                sig_off()
        if self._type == nr_dpe:
            sig_on()
            self._core.dpe.optimize_coefficients_cost_fixed_prob(pr_)
            called = True
            sig_off()
        IF HAVE_QD:
            if self._type == nr_dd:
                sig_on()
                self._core.dd.optimize_coefficients_cost_fixed_prob(pr_)
                called = True
                sig_off()
            elif self._type == nr_qd:
                sig_on()
                self._core.qd.optimize_coefficients_cost_fixed_prob(pr_)
                called = True
                sig_off()
        if self._type == nr_mpfr:
            sig_on()
            self._core.mpfr.optimize_coefficients_cost_fixed_prob(pr_)
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
        Compute the cost of a single enumeration::

            >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
            >>> _ = LLL.reduction(A)
            >>> M = GSO.Mat(A)
            >>> _ = M.update_gso()
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 0.51)
            >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
            >>> cost, details = pr.single_enum_cost(c, True)
            >>> cost  # doctest: +ELLIPSIS
            14980.48...

            >>> details[0:10]  # doctest: +ELLIPSIS
            (0.134901..., 0.3048..., 0.81588..., 1.945..., 4.5903..., 11.51..., 16.048..., 41.7115..., 48.03..., 116.986...)

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
        retrials to reach target::

            >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
            >>> _ = LLL.reduction(A)
            >>> M = GSO.Mat(A)
            >>> _ = M.update_gso()
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 0.51)
            >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])
            >>> pr.repeated_enum_cost(c)  # doctest: +ELLIPSIS
            15626.98...

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
        Compute the success probability of expected number of solutions of a single enumeration::

            >>> from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
            >>> FPLLL.set_random_seed(1337)
            >>> A = IntegerMatrix.random(40, "qary", bits=20, k=20)
            >>> _ = LLL.reduction(A)
            >>> M = GSO.Mat(A)
            >>> _ = M.update_gso()
            >>> pr = Pruning.Pruner(M.get_r(0,0), 2**20, [M.r()], 0.51)
            >>> c = pr.optimize_coefficients([1. for _ in range(M.d)])

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

def prune(double enumeration_radius, double preproc_cost, gso_r, double target,
          metric="probability", int flags=PRUNER_GRADIENT, pruning=None, float_type="double"):
    """Return optimal pruning parameters.

    :param enumeration_radius: target squared enumeration radius
    :param preproc_cost: cost of preprocessing
    :param gso_: list (of lists) with r coefficients
    :param target: overall targeted success probability or number of solutions
    :param metric: "probability" or "solutions"
    :param flags: flags
    :param pruning: write output here, pass ``None`` for creating a new one
    :param float_type: floating point type to use

    EXAMPLE::

        >>> from fpylll import IntegerMatrix, LLL, GSO, FPLLL
        >>> from fpylll import FPLLL
        >>> from fpylll import Pruning
        >>> FPLLL.set_random_seed(1337)
        >>> A = IntegerMatrix.random(20, "qary", bits=20, k=10)
        >>> M = GSO.Mat(A)
        >>> LLL.Reduction(M)()
        >>> _ = FPLLL.set_precision(128)
        >>> R = [M.get_r(i,i) for i in range(0, 20)]
        >>> pr0 = Pruning.run(R[0], 2**20, [R], 0.5, float_type="double")
        >>> pr1 = Pruning.run(R[0], 2**20, [R], 0.5, float_type="mpfr")

        >>> pr0.coefficients[10], pr1.coefficients[10] # doctest: +ELLIPSIS
        (0.6266..., 0.6266...)

        >>> pr0 = Pruning.run(R[0], 2**10, [R], 0.5, flags=Pruning.GRADIENT, float_type="double")
        >>> pr1 = Pruning.run(R[0], 2**10, [R], 0.5, flags=Pruning.NELDER_MEAD, float_type="mpfr")
        >>> pr0.coefficients[10], pr1.coefficients[10] # doctest: +ELLIPSIS
        (0.70722482938..., 0.824291475...)

    ..  note :: Preprocessing cost should be expressed in terms of nodes in an enumeration (~100
        CPU cycles per node)
    """
    if preproc_cost < 1:
        raise ValueError("Preprocessing cost must be at least 1 but got %f"%preproc_cost)
    if metric == PRUNER_METRIC_PROBABILITY_OF_SHORTEST:
        if target <= 0 or target >= 1.0:
            raise ValueError("Probability must be between 0 and 1 (exclusive) but got %f"%target)
    if metric == PRUNER_METRIC_EXPECTED_SOLUTIONS:
        if target <= 0:
            raise ValueError("Number of solutions must be > 0 but got %f"%target)

    cdef FloatType ft = check_float_type(float_type)
    metric = check_pruner_metric(metric)

    try:
        gso_r[0][0]
    except (AttributeError, TypeError):
        gso_r = [gso_r]

    if pruning is None:
        pruning = PruningParams(1.0, [], 1.0)
    elif not isinstance(pruning, PruningParams):
        raise TypeError("First parameter must be of type PruningParams or None but got type '%s'"%type(pruning))

    cdef vector[vector[double]] gso_r_

    d = len(gso_r[0])
    for i,m in enumerate(gso_r):
        gso_r_.push_back(vector[double]())
        if len(m) != d:
            raise ValueError("Lengths of all vectors must match.")
        for e in m:
            gso_r_[i].push_back(e)

    if ft == FT_DOUBLE:
        sig_on()
        prune_c[FP_NR[double]]((<PruningParams>pruning)._core, enumeration_radius, preproc_cost,
                               gso_r_, target, metric, flags)
        sig_off()
        return pruning
    IF HAVE_LONG_DOUBLE:
        if ft == FT_LONG_DOUBLE:
            sig_on()
            prune_c[FP_NR[longdouble]]((<PruningParams>pruning)._core, enumeration_radius, preproc_cost,
                                       gso_r_, target, metric, flags)
            sig_off()
            return pruning
    if ft == FT_DPE:
        sig_on()
        prune_c[FP_NR[dpe_t]]((<PruningParams>pruning)._core, enumeration_radius, preproc_cost,
                              gso_r_, target, metric, flags)
        sig_off()
        return pruning
    if ft == FT_MPFR:
        sig_on()
        prune_c[FP_NR[mpfr_t]]((<PruningParams>pruning)._core, enumeration_radius, preproc_cost,
                               gso_r_, target, metric, flags)
        sig_off()
        return pruning
    IF HAVE_QD:
            if ft == FT_DD:
                sig_on()
                prune_c[FP_NR[dd_t]]((<PruningParams>pruning)._core, enumeration_radius, preproc_cost,
                                     gso_r_, target, metric, flags)
                sig_off()
                return pruning
            elif ft == FT_QD:
                sig_on()
                prune_c[FP_NR[qd_t]]((<PruningParams>pruning)._core, enumeration_radius, preproc_cost,
                                     gso_r_, target, metric, flags)
                sig_off()
                return pruning


def svp_probability(pr, float_type="double"):
    """Return probability of success for enumeration with given set of pruning parameters.

    :param pr: pruning parameters, either PruningParams object or list of floating point numbers
    :param float_type: floating point type used internally

    """
    cdef FloatType ft = check_float_type(float_type)

    if not isinstance(pr, PruningParams):
        pr = PruningParams(1.0, pr, 1.0)

    if ft == FT_DOUBLE:
        return svp_probability_c[FP_NR[double]]((<PruningParams>pr)._core.coefficients).get_d()
    IF HAVE_LONG_DOUBLE:
        if ft == FT_LONG_DOUBLE:
            return svp_probability_c[FP_NR[longdouble]]((<PruningParams>pr)._core.coefficients).get_d()
    if ft == FT_DPE:
        return svp_probability_c[FP_NR[dpe_t]]((<PruningParams>pr)._core.coefficients).get_d()
    if ft == FT_MPFR:
        return svp_probability_c[FP_NR[mpfr_t]]((<PruningParams>pr)._core.coefficients).get_d()
    IF HAVE_QD:
        if ft == FT_DD:
            return svp_probability_c[FP_NR[dd_t]]((<PruningParams>pr)._core.coefficients).get_d()
        elif ft == FT_QD:
            return svp_probability_c[FP_NR[qd_t]]((<PruningParams>pr)._core.coefficients).get_d()

    raise ValueError("Float type '%s' not understood."%float_type)

class Pruning:
    Pruner = Pruner
    PruningParams = PruningParams
    LinearPruningParams = PruningParams.LinearPruningParams
    run = staticmethod(prune)

    CVP = PRUNER_CVP
    START_FROM_INPUT = PRUNER_START_FROM_INPUT
    GRADIENT = PRUNER_GRADIENT
    NELDER_MEAD = PRUNER_NELDER_MEAD
    VERBOSE = PRUNER_VERBOSE
    ZEALOUS = PRUNER_GRADIENT | PRUNER_NELDER_MEAD
    SINGLE = PRUNER_SINGLE
    HALF = PRUNER_HALF
    PROBABILITY_OF_SHORTEST = PRUNER_METRIC_PROBABILITY_OF_SHORTEST
    EXPECTED_SOLUTIONS = PRUNER_METRIC_EXPECTED_SOLUTIONS
