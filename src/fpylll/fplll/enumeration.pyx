# -*- coding: utf-8 -*-
include "fpylll/config.pxi"

from cython.operator cimport dereference as deref, preincrement as inc

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from cysignals.signals cimport sig_on, sig_off

from .fplll cimport EvaluatorStrategy as EvaluatorStrategy_c
from .fplll cimport EVALSTRATEGY_BEST_N_SOLUTIONS
from .fplll cimport EVALSTRATEGY_FIRST_N_SOLUTIONS
from .fplll cimport EVALSTRATEGY_OPPORTUNISTIC_N_SOLUTIONS
from .fplll cimport Enumeration as Enumeration_c
from .fplll cimport FastEvaluator as FastEvaluator_c
from .fplll cimport CallbackEvaluator as CallbackEvaluator_c
from .fplll cimport Evaluator as Evaluator_c
from .fplll cimport FastErrorBoundedEvaluator as FastErrorBoundedEvaluator_c
from .fplll cimport ErrorBoundedEvaluator as ErrorBoundedEvaluator_c
from .fplll cimport MatGSOInterface as MatGSOInterface_c
from .fplll cimport Z_NR, FP_NR, mpz_t
from .fplll cimport EVALMODE_SV

from .fplll cimport dpe_t
from fpylll.mpfr.mpfr cimport mpfr_t
from .decl cimport mat_gso_mpz_d, mat_gso_mpz_ld, mat_gso_mpz_dpe, mat_gso_mpz_mpfr, fp_nr_t
from .decl cimport mat_gso_long_d, mat_gso_long_ld, mat_gso_long_dpe, mat_gso_long_mpfr
from .decl cimport mat_gso_gso_t, mat_gso_gram_t
from .decl cimport d_t
from .fplll cimport FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR, FloatType

from .fplll cimport multimap

from .fplll cimport FPLLL_MAX_ENUM_DIM

from libcpp cimport bool

cdef public bool evaluator_callback_call_obj(obj, int n, double *new_sol_coord):
    cdef list new_sol_coord_ = []
    for i in range(n):
        new_sol_coord_.append(new_sol_coord[i])
    return obj(new_sol_coord_);

IF HAVE_LONG_DOUBLE:
    from .decl cimport ld_t

IF HAVE_QD:
    from .decl cimport mat_gso_mpz_dd, mat_gso_mpz_qd, mat_gso_long_dd, mat_gso_long_qd, dd_t, qd_t
    from .fplll cimport FT_DD, FT_QD

class EnumerationError(Exception):
    pass

class EvaluatorStrategy:
    """
    Strategies to update the enumeration radius and deal with multiple solutions.  Possible values
    are:

    - ``BEST_N_SOLUTIONS`` Starting with the nr_solutions-th solution, every time a new solution is
      found the enumeration bound is updated to the length of the longest solution.  If more
      than nr_solutions were found, the longest is dropped.

    - ``OPPORTUNISTIC_N_SOLUTIONS`` Every time a solution is found, update the enumeration distance
      to the length of the solution.  If more than nr_solutions were found, the longest is
      dropped.

    - ``FIRST_N_SOLUTIONS`` The enumeration bound is not updated.  As soon as nr_solutions are
      found, enumeration stops.
    """
    BEST_N_SOLUTIONS = EVALSTRATEGY_BEST_N_SOLUTIONS
    OPPORTUNISTIC_N_SOLUTIONS = EVALSTRATEGY_OPPORTUNISTIC_N_SOLUTIONS
    FIRST_N_SOLUTIONS = EVALSTRATEGY_FIRST_N_SOLUTIONS


cdef class Enumeration:
    def __cinit__(self, MatGSO M, int nr_solutions=1,
                  strategy=EvaluatorStrategy.BEST_N_SOLUTIONS, bool sub_solutions=False,
                  callbackf=None):
        """Create new enumeration object

        :param MatGSO M:       GSO matrix
        :param nr_solutions:   Number of solutions to be returned by enumeration
        :param strategy:       EvaluatorStrategy to use when finding new solutions
        :param sub_solutions:  Compute sub-solutions
        :param callbackf:      A predicate to accept or reject a candidate solution

        """

        cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[d_t]]  *m_mpz_d
        IF HAVE_LONG_DOUBLE:
            cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[ld_t]] *m_mpz_ld
        cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dpe_t]] *m_mpz_dpe
        IF HAVE_QD:
            cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dd_t]] *m_mpz_dd
            cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[qd_t]] *m_mpz_qd
        cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[mpfr_t]]  *m_mpz_mpfr

        cdef MatGSOInterface_c[Z_NR[long], FP_NR[d_t]]  *m_l_d
        IF HAVE_LONG_DOUBLE:
            cdef MatGSOInterface_c[Z_NR[long], FP_NR[ld_t]] *m_l_ld
        cdef MatGSOInterface_c[Z_NR[long], FP_NR[dpe_t]] *m_l_dpe
        IF HAVE_QD:
            cdef MatGSOInterface_c[Z_NR[long], FP_NR[dd_t]] *m_l_dd
            cdef MatGSOInterface_c[Z_NR[long], FP_NR[qd_t]] *m_l_qd
        cdef MatGSOInterface_c[Z_NR[long], FP_NR[mpfr_t]]  *m_l_mpfr

        self.M = M

        if M._type == mat_gso_mpz_d:
            m_mpz_d = <MatGSOInterface_c[Z_NR[mpz_t], FP_NR[d_t]]*>M._core.mpz_d
            if callbackf is None:
                self._eval_core.d = <Evaluator_c[FP_NR[double]]*>new FastEvaluator_c[FP_NR[double]](nr_solutions,
                                                                                                  strategy,
                                                                                                  sub_solutions)
            else:
                self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                self._eval_core.d = <Evaluator_c[FP_NR[double]]*>new CallbackEvaluator_c[FP_NR[double]](
                    self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
            self._core.mpz_d = new Enumeration_c[Z_NR[mpz_t], FP_NR[double]](m_mpz_d[0], self._eval_core.d[0])
        elif M._type == mat_gso_long_d:
            m_l_d = <MatGSOInterface_c[Z_NR[long], FP_NR[d_t]]*>M._core.long_d
            if callbackf is None:
                self._eval_core.d = <Evaluator_c[FP_NR[d_t]]*>new FastEvaluator_c[FP_NR[d_t]](nr_solutions,
                                                                                                  strategy,
                                                                                                  sub_solutions)
            else:
                self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                self._eval_core.d = <Evaluator_c[FP_NR[d_t]]*>new CallbackEvaluator_c[FP_NR[d_t]](
                    self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
            self._core.long_d = new Enumeration_c[Z_NR[long], FP_NR[double]](m_l_d[0], self._eval_core.d[0])
        elif M._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                m_mpz_ld =  <MatGSOInterface_c[Z_NR[mpz_t], FP_NR[longdouble]]*>M._core.mpz_ld
                if callbackf is None:
                    self._eval_core.ld = <Evaluator_c[FP_NR[ld_t]]*>new FastEvaluator_c[FP_NR[ld_t]](nr_solutions,
                                                                                                      strategy,
                                                                                                      sub_solutions)
                else:
                    self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                    self._eval_core.ld = <Evaluator_c[FP_NR[ld_t]]*>new CallbackEvaluator_c[FP_NR[ld_t]](
                        self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
                self._core.mpz_ld = new Enumeration_c[Z_NR[mpz_t], FP_NR[ld_t]](m_mpz_ld[0], self._eval_core.ld[0])
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)
        elif M._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                m_l_ld = <MatGSOInterface_c[Z_NR[long], FP_NR[ld_t]]*>M._core.long_ld
                if callbackf is None:
                    self._eval_core.ld = <Evaluator_c[FP_NR[ld_t]]*>new FastEvaluator_c[FP_NR[ld_t]](nr_solutions,
                                                                                                      strategy,
                                                                                                      sub_solutions)
                else:
                    self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                    self._eval_core.ld = <Evaluator_c[FP_NR[ld_t]]*>new CallbackEvaluator_c[FP_NR[ld_t]](
                        self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
                self._core.long_ld = new Enumeration_c[Z_NR[long], FP_NR[ld_t]](m_l_ld[0], self._eval_core.ld[0])
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)
        elif M._type == mat_gso_mpz_dpe:
            m_mpz_dpe = <MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dpe_t]]*>M._core.mpz_dpe
            if callbackf is None:
                self._eval_core.dpe = <Evaluator_c[FP_NR[dpe_t]]*>new FastEvaluator_c[FP_NR[dpe_t]](nr_solutions,
                                                                                                  strategy,
                                                                                                  sub_solutions)
            else:
                self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                self._eval_core.dpe = <Evaluator_c[FP_NR[dpe_t]]*>new CallbackEvaluator_c[FP_NR[dpe_t]](
                    self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
            self._core.mpz_dpe = new Enumeration_c[Z_NR[mpz_t], FP_NR[dpe_t]](m_mpz_dpe[0], self._eval_core.dpe[0])
        elif M._type == mat_gso_long_dpe:
            m_l_dpe = <MatGSOInterface_c[Z_NR[long], FP_NR[dpe_t]]*>M._core.long_dpe
            if callbackf is None:
                self._eval_core.dpe = <Evaluator_c[FP_NR[dpe_t]]*>new FastEvaluator_c[FP_NR[dpe_t]](nr_solutions,
                                                                                                  strategy,
                                                                                                  sub_solutions)
            else:
                self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                self._eval_core.dpe = <Evaluator_c[FP_NR[dpe_t]]*>new CallbackEvaluator_c[FP_NR[dpe_t]](
                    self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
            self._core.long_dpe = new Enumeration_c[Z_NR[long], FP_NR[dpe_t]](m_l_dpe[0], self._eval_core.dpe[0])
        elif M._type == mat_gso_mpz_mpfr:
            if callbackf is not None:
                raise NotImplementedError("Callbacks are not implemented for MPFR.")
            m_mpz_mpfr = <MatGSOInterface_c[Z_NR[mpz_t], FP_NR[mpfr_t]]*>M._core.mpz_mpfr
            self._eval_core.mpfr = <ErrorBoundedEvaluator_c*>new FastErrorBoundedEvaluator_c(M.d,
                                                                                             M._core.mpz_mpfr.get_mu_matrix(),
                                                                                             M._core.mpz_mpfr.get_r_matrix(),
                                                                                             EVALMODE_SV,
                                                                                             nr_solutions,
                                                                                             strategy,
                                                                                             sub_solutions)
            self._core.mpz_mpfr = new Enumeration_c[Z_NR[mpz_t], FP_NR[mpfr_t]](m_mpz_mpfr[0], self._eval_core.mpfr[0])
        elif M._type == mat_gso_long_mpfr:
            if callbackf is not None:
                raise NotImplementedError("Callbacks are not implemented for MPFR.")
            m_l_mpfr = <MatGSOInterface_c[Z_NR[long], FP_NR[mpfr_t]]*>M._core.long_mpfr
            self._eval_core.mpfr = <ErrorBoundedEvaluator_c*>new FastErrorBoundedEvaluator_c(M.d,
                                                                                           M._core.long_mpfr.get_mu_matrix(),
                                                                                           M._core.long_mpfr.get_r_matrix(),
                                                                                           EVALMODE_SV,
                                                                                           nr_solutions,
                                                                                           strategy,
                                                                                           sub_solutions)
            self._core.long_mpfr = new Enumeration_c[Z_NR[long], FP_NR[mpfr_t]](m_l_mpfr[0], self._eval_core.mpfr[0])
        else:
            IF HAVE_QD:
                if M._type == mat_gso_mpz_dd:
                    m_mpz_dd = <MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dd_t]]*>M._core.mpz_dd
                    if callbackf is None:
                        self._eval_core.dd = <Evaluator_c[FP_NR[dd_t]]*>new FastEvaluator_c[FP_NR[dd_t]](nr_solutions,
                                                                                                          strategy,
                                                                                                          sub_solutions)
                    else:
                        self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                        self._eval_core.dd = <Evaluator_c[FP_NR[dd_t]]*>new CallbackEvaluator_c[FP_NR[dd_t]](
                            self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
                    self._core.mpz_dd = new Enumeration_c[Z_NR[mpz_t], FP_NR[dd_t]](m_mpz_dd[0], self._eval_core.dd[0])
                elif M._type == mat_gso_mpz_qd:
                    m_mpz_qd = <MatGSOInterface_c[Z_NR[mpz_t], FP_NR[qd_t]]*>M._core.mpz_qd
                    if callbackf is None:
                        self._eval_core.qd = <Evaluator_c[FP_NR[qd_t]]*>new FastEvaluator_c[FP_NR[qd_t]](nr_solutions,
                                                                                                          strategy,
                                                                                                          sub_solutions)
                    else:
                        self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                        self._eval_core.qd = <Evaluator_c[FP_NR[qd_t]]*>new CallbackEvaluator_c[FP_NR[qd_t]](
                            self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
                    self._core.mpz_qd = new Enumeration_c[Z_NR[mpz_t], FP_NR[qd_t]](m_mpz_qd[0], self._eval_core.qd[0])
                elif M._type == mat_gso_long_dd:
                    m_l_dd = <MatGSOInterface_c[Z_NR[long], FP_NR[dd_t]]*>M._core.long_dd
                    if callbackf is None:
                        self._eval_core.dd = <Evaluator_c[FP_NR[dd_t]]*>new FastEvaluator_c[FP_NR[dd_t]](nr_solutions,
                                                                                                          strategy,
                                                                                                          sub_solutions)
                    else:
                        self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                        self._eval_core.dd = <Evaluator_c[FP_NR[dd_t]]*>new CallbackEvaluator_c[FP_NR[dd_t]](
                            self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
                    self._core.long_dd = new Enumeration_c[Z_NR[long], FP_NR[dd_t]](m_l_dd[0], self._eval_core.dd[0])
                elif M._type == mat_gso_long_qd:
                    m_l_qd = <MatGSOInterface_c[Z_NR[long], FP_NR[qd_t]]*>M._core.long_qd
                    if callbackf is None:
                        self._eval_core.qd = <Evaluator_c[FP_NR[qd_t]]*>new FastEvaluator_c[FP_NR[qd_t]](nr_solutions,
                                                                                                          strategy,
                                                                                                          sub_solutions)
                    else:
                        self._callback_wrapper = new PyCallbackEvaluatorWrapper_c(callbackf)
                        self._eval_core.qd = <Evaluator_c[FP_NR[qd_t]]*>new CallbackEvaluator_c[FP_NR[qd_t]](
                            self._callback_wrapper[0], NULL, nr_solutions, strategy, sub_solutions)
                    self._core.long_qd = new Enumeration_c[Z_NR[long], FP_NR[qd_t]](m_l_qd[0], self._eval_core.qd[0])
                else:
                    raise RuntimeError("MatGSO object '%s' has no core."%self)
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)

    def __dealloc__(self):
        if self.M._type == mat_gso_mpz_d:
            del self._eval_core.d
            del self._core.mpz_d
        IF HAVE_LONG_DOUBLE:
            if self.M._type == mat_gso_mpz_ld:
                del self._eval_core.ld
                del self._core.mpz_ld
        if self.M._type == mat_gso_mpz_dpe:
            del self._eval_core.dpe
            del self._core.mpz_dpe
        IF HAVE_QD:
            if self.M._type == mat_gso_mpz_dd:
                del self._eval_core.dd
                del self._core.mpz_dd
            if self.M._type == mat_gso_mpz_qd:
                del self._eval_core.qd
                del self._core.mpz_qd
        if self.M._type == mat_gso_mpz_mpfr:
            del self._eval_core.mpfr
            del self._core.mpz_mpfr
        if self.M._type == mat_gso_long_d:
            del self._eval_core.d
            del self._core.long_d
        IF HAVE_LONG_DOUBLE:
            if self.M._type == mat_gso_long_ld:
                del self._eval_core.ld
                del self._core.long_ld
        if self.M._type == mat_gso_long_dpe:
            del self._eval_core.dpe
            del self._core.long_dpe
        IF HAVE_QD:
            if self.M._type == mat_gso_long_dd:
                del self._eval_core.dd
                del self._core.long_dd
            if self.M._type == mat_gso_long_qd:
                del self._eval_core.qd
                del self._core.long_qd
        if self.M._type == mat_gso_long_mpfr:
            del self._eval_core.mpfr
            del self._core.long_mpfr

        if self._callback_wrapper:
            del self._callback_wrapper

    def enumerate(self, int first, int last, max_dist, max_dist_expo,
                  target=None, subtree=None, pruning=None, dual=False, subtree_reset=False):
        """Run enumeration on `M`

        :param int first:      first row
        :param int last:       last row (exclusive)
        :param max_dist:       length bound
        :param max_dist_expo:  exponent of length bound
        :param target:         target coordinates for CVP/BDD or ``None`` for SVP
        :param subtree:
        :param pruning:        pruning parameters
        :param dual:           run enumeration in the primal or dual lattice.
        :param subtree_reset:
        :returns: list of pairs containing the solutions' coefficient vectors and their lengths

        """
        cdef int block_size = last-first
        cdef fp_nr_t tmp

        cdef vector[FP_NR[d_t]] target_coord_d
        IF HAVE_LONG_DOUBLE:
            cdef vector[FP_NR[ld_t]] target_coord_ld
        cdef vector[FP_NR[dpe_t]] target_coord_dpe
        IF HAVE_QD:
            cdef vector[FP_NR[dd_t]] target_coord_dd
            cdef vector[FP_NR[qd_t]] target_coord_qd
        cdef vector[FP_NR[mpfr_t]] target_coord_mpfr

        cdef vector[double] sub_tree_

        if subtree is not None:
            for it in target:
                sub_tree_.push_back(float(it))

        cdef vector[double] pruning_

        if not pruning:
            for i in range(block_size):
                pruning_.push_back(1)
        else:
            for i in range(block_size):
                pruning_.push_back(pruning[i])

        cdef double max_dist__ = max_dist
        cdef FP_NR[d_t] max_dist_d = max_dist__
        IF HAVE_LONG_DOUBLE:
            cdef FP_NR[ld_t] max_dist_ld = max_dist__
        cdef FP_NR[dpe_t] max_dist_dpe = max_dist__
        IF HAVE_QD:
            cdef FP_NR[dd_t] max_dist_dd = max_dist__
            cdef FP_NR[qd_t] max_dist_qd = max_dist__
        cdef FP_NR[mpfr_t] max_dist_mpfr = max_dist__

        solutions = []
        cdef multimap[FP_NR[double], vector[FP_NR[double]]].reverse_iterator solutions_d
        IF HAVE_LONG_DOUBLE:
            cdef multimap[FP_NR[longdouble], vector[FP_NR[longdouble]]].reverse_iterator solutions_ld
        cdef multimap[FP_NR[dpe_t], vector[FP_NR[dpe_t]]].reverse_iterator solutions_dpe
        IF HAVE_QD:
            cdef multimap[FP_NR[dd_t], vector[FP_NR[dd_t]]].reverse_iterator solutions_dd
            cdef multimap[FP_NR[qd_t], vector[FP_NR[qd_t]]].reverse_iterator solutions_qd
        cdef multimap[FP_NR[mpfr_t], vector[FP_NR[mpfr_t]]].reverse_iterator solutions_mpfr

        if self.M._type == mat_gso_mpz_d or self.M._type == mat_gso_long_d:
            if target is not None:
                for it in target:
                    tmp.d = float(it)
                    target_coord_d.push_back(tmp.d)
            sig_on()
            if self.M._type == mat_gso_mpz_d:
                self._core.mpz_d.enumerate(first, last, max_dist_d, max_dist_expo,
                                           target_coord_d, sub_tree_, pruning_, dual)
            else:
                self._core.long_d.enumerate(first, last, max_dist_d, max_dist_expo,
                                            target_coord_d, sub_tree_, pruning_, dual)
            sig_off()
            if not self._eval_core.d.size():
                raise EnumerationError("No solution found.")

            solutions_d = self._eval_core.d.begin()
            while solutions_d != self._eval_core.d.end():
                cur_dist = deref(solutions_d).first.get_d()
                cur_sol = []
                for j in range(deref(solutions_d).second.size()):
                    cur_sol.append(deref(solutions_d).second[j].get_d())
                solutions.append((cur_dist, tuple(cur_sol)))
                inc(solutions_d)

        IF HAVE_LONG_DOUBLE:
            if self.M._type == mat_gso_mpz_ld or self.M._type == mat_gso_long_ld:
                if target is not None:
                    for it in target:
                        tmp.ld = float(it)
                        target_coord_ld.push_back(tmp.ld)
                sig_on()
                if self.M._type == mat_gso_mpz_ld:
                    self._core.mpz_ld.enumerate(first, last, max_dist_ld, max_dist_expo,
                                                target_coord_ld, sub_tree_, pruning_, dual)
                else:
                    self._core.long_ld.enumerate(first, last, max_dist_ld, max_dist_expo,
                                                 target_coord_ld, sub_tree_, pruning_, dual)
                sig_off()
                if not self._eval_core.ld.size():
                    raise EnumerationError("No solution found.")

                solutions_ld = self._eval_core.ld.begin()
                while solutions_ld != self._eval_core.ld.end():
                    cur_dist = deref(solutions_ld).first.get_d()
                    cur_sol = []
                    for j in range(deref(solutions_ld).second.size()):
                        cur_sol.append(deref(solutions_ld).second[j].get_d())
                    solutions.append((cur_dist, tuple(cur_sol)))
                    inc(solutions_ld)

        if self.M._type == mat_gso_mpz_dpe or self.M._type == mat_gso_long_dpe:
            if target is not None:
                for it in target:
                    tmp.dpe = float(it)
                    target_coord_dpe.push_back(tmp.dpe)
            sig_on()
            if self.M._type == mat_gso_mpz_dpe:
                self._core.mpz_dpe.enumerate(first, last, max_dist_dpe, max_dist_expo,
                                           target_coord_dpe, sub_tree_, pruning_, dual)
            else:
                self._core.long_dpe.enumerate(first, last, max_dist_dpe, max_dist_expo,
                                            target_coord_dpe, sub_tree_, pruning_, dual)
            sig_off()
            if not self._eval_core.dpe.size():
                raise EnumerationError("No solution found.")

            solutions_dpe = self._eval_core.dpe.begin()
            while solutions_dpe != self._eval_core.dpe.end():
                cur_dist = deref(solutions_dpe).first.get_d()
                cur_sol = []
                for j in range(deref(solutions_dpe).second.size()):
                    cur_sol.append(deref(solutions_dpe).second[j].get_d())
                solutions.append((cur_dist, tuple(cur_sol)))
                inc(solutions_dpe)

        IF HAVE_QD:
            if self.M._type == mat_gso_mpz_dd or self.M._type == mat_gso_long_dd:
                if target is not None:
                    for it in target:
                        tmp.dd = float(it)
                        target_coord_dd.push_back(tmp.dd)
                sig_on()
                if self.M._type == mat_gso_mpz_dd:
                    self._core.mpz_dd.enumerate(first, last, max_dist_dd, max_dist_expo,
                                               target_coord_dd, sub_tree_, pruning_, dual)
                else:
                    self._core.long_dd.enumerate(first, last, max_dist_dd, max_dist_expo,
                                                target_coord_dd, sub_tree_, pruning_, dual)
                sig_off()
                if not self._eval_core.dd.size():
                    raise EnumerationError("No solution found.")

                solutions_dd = self._eval_core.dd.begin()
                while solutions_dd != self._eval_core.dd.end():
                    cur_dist = deref(solutions_dd).first.get_d()
                    cur_sol = []
                    for j in range(deref(solutions_dd).second.size()):
                        cur_sol.append(deref(solutions_dd).second[j].get_d())
                    solutions.append((cur_dist, tuple(cur_sol)))
                    inc(solutions_dd)

            if self.M._type == mat_gso_mpz_qd or self.M._type == mat_gso_long_qd:
                if target is not None:
                    for it in target:
                        tmp.qd = float(it)
                        target_coord_qd.push_back(tmp.qd)
                sig_on()
                if self.M._type == mat_gso_mpz_qd:
                    self._core.mpz_qd.enumerate(first, last, max_dist_qd, max_dist_expo,
                                               target_coord_qd, sub_tree_, pruning_, dual)
                else:
                    self._core.long_qd.enumerate(first, last, max_dist_qd, max_dist_expo,
                                                target_coord_qd, sub_tree_, pruning_, dual)
                sig_off()
                if not self._eval_core.qd.size():
                    raise EnumerationError("No solution found.")

                solutions_qd = self._eval_core.qd.begin()
                while solutions_qd != self._eval_core.qd.end():
                    cur_dist = deref(solutions_qd).first.get_d()
                    cur_sol = []
                    for j in range(deref(solutions_qd).second.size()):
                        cur_sol.append(deref(solutions_qd).second[j].get_d())
                    solutions.append((cur_dist, tuple(cur_sol)))
                    inc(solutions_qd)

        if self.M._type == mat_gso_mpz_mpfr or self.M._type == mat_gso_long_mpfr:
            if target is not None:
                for it in target:
                    tmp.mpfr = float(it)
                    target_coord_mpfr.push_back(tmp.mpfr)
            sig_on()
            if self.M._type == mat_gso_mpz_mpfr:
                self._core.mpz_mpfr.enumerate(first, last, max_dist_mpfr, max_dist_expo,
                                           target_coord_mpfr, sub_tree_, pruning_, dual)
            else:
                self._core.long_mpfr.enumerate(first, last, max_dist_mpfr, max_dist_expo,
                                            target_coord_mpfr, sub_tree_, pruning_, dual)
            sig_off()
            if not self._eval_core.mpfr.size():
                raise EnumerationError("No solution found.")

            solutions_mpfr = self._eval_core.mpfr.begin()
            while solutions_mpfr != self._eval_core.mpfr.end():
                cur_dist = deref(solutions_mpfr).first.get_d()
                cur_sol = []
                for j in range(deref(solutions_mpfr).second.size()):
                    cur_sol.append(deref(solutions_mpfr).second[j].get_d())
                solutions.append((cur_dist, tuple(cur_sol)))
                inc(solutions_mpfr)

        if solutions == []:
            raise NotImplementedError("GSO datatype not implemented.")

        return solutions

    @property
    def sub_solutions(self):
        """
        Return sub-solutions computed in last enumeration call.

        >>> from fpylll import *
        >>> FPLLL.set_random_seed(1337)
        >>> _ = FPLLL.set_threads(1)
        >>> A = IntegerMatrix.random(80, "qary", bits=30, k=40)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A)
        >>> _ = M.update_gso()
        >>> pruning = Pruning.run(M.get_r(0, 0), 2**40, M.r()[:30], 0.8)
        >>> enum = Enumeration(M, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS, sub_solutions=True)
        >>> _ = enum.enumerate(0, 30, 0.999*M.get_r(0, 0), 0, pruning=pruning.coefficients)
        >>> [int(round(a)) for a,b in enum.sub_solutions[:5]]
        [13018980230, 12980748618, 12469398480, 10737191842, 10723577014]

        """
        cdef list sub_solutions = []

        cdef vector[pair[FP_NR[d_t], vector[FP_NR[d_t]]]].iterator _sub_solutions_d

        if self.M._type == mat_gso_mpz_d or self.M._type == mat_gso_long_d:
            _sub_solutions_d = self._eval_core.d.sub_solutions.begin()
            while _sub_solutions_d != self._eval_core.d.sub_solutions.end():
                cur_dist = deref(_sub_solutions_d).first.get_d()
                if cur_dist == 0.0:
                    cur_dist = None
                cur_sol = []
                for j in range(deref(_sub_solutions_d).second.size()):
                    cur_sol.append(deref(_sub_solutions_d).second[j].get_d())
                sub_solutions.append(tuple([cur_dist, tuple(cur_sol)]))
                inc(_sub_solutions_d)
        else:
            raise NotImplementedError

        return tuple(sub_solutions)

    def get_nodes(self, level=None):
        """
        Return number of visited nodes in last enumeration call.

        :param level: return for ``level`` except when ``None`` in which case the sum is returned.
        """

        cdef int _level = -1

        if level is not None:
            if level < -1 or level >= FPLLL_MAX_ENUM_DIM:
                raise ValueError("Level {level} out of bounds.".format(level=level))
            _level = level

        if self.M._type == mat_gso_mpz_d:
            return self._core.mpz_d.get_nodes(_level)
        IF HAVE_LONG_DOUBLE:
            if self.M._type == mat_gso_mpz_ld:
                return self._core.mpz_ld.get_nodes(_level)
        if self.M._type == mat_gso_mpz_dpe:
            return self._core.mpz_dpe.get_nodes(_level)
        IF HAVE_QD:
            if self.M._type == mat_gso_mpz_dd:
                return self._core.mpz_dd.get_nodes(_level)
            if self.M._type == mat_gso_mpz_qd:
                return self._core.mpz_qd.get_nodes(_level)
        if self.M._type == mat_gso_mpz_mpfr:
            return self._core.mpz_mpfr.get_nodes(_level)

        if self.M._type == mat_gso_long_d:
            return self._core.long_d.get_nodes(_level)
        IF HAVE_LONG_DOUBLE:
            if self.M._type == mat_gso_long_ld:
                return self._core.long_ld.get_nodes(_level)
        if self.M._type == mat_gso_long_dpe:
            return self._core.long_dpe.get_nodes(_level)
        IF HAVE_QD:
            if self.M._type == mat_gso_long_dd:
                return self._core.long_dd.get_nodes(_level)
            if self.M._type == mat_gso_long_qd:
                return self._core.long_qd.get_nodes(_level)
        if self.M._type == mat_gso_long_mpfr:
            return self._core.long_mpfr.get_nodes(_level)
