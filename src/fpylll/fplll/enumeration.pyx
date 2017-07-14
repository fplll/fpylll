# -*- coding: utf-8 -*-
include "fpylll/config.pxi"

from cython.operator cimport dereference as deref, preincrement as inc

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
from cysignals.signals cimport sig_on, sig_off

from gso cimport MatGSO
from fplll cimport EvaluatorStrategy as EvaluatorStrategy_c
from fplll cimport EVALSTRATEGY_BEST_N_SOLUTIONS
from fplll cimport EVALSTRATEGY_FIRST_N_SOLUTIONS
from fplll cimport EVALSTRATEGY_OPPORTUNISTIC_N_SOLUTIONS
from fplll cimport Enumeration as Enumeration_c
from fplll cimport FastEvaluator as FastEvaluator_c
from fplll cimport FastErrorBoundedEvaluator as FastErrorBoundedEvaluator_c
from fplll cimport MatGSO as MatGSO_c
from fplll cimport Z_NR, FP_NR, mpz_t
from fplll cimport EVALMODE_SV

from fplll cimport dpe_t
from fpylll.mpfr.mpfr cimport mpfr_t
from decl cimport gso_mpz_d, gso_mpz_ld, gso_mpz_dpe, gso_mpz_mpfr, fp_nr_t
from decl cimport d_t, ld_t
from fplll cimport FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR, FloatType

from fplll cimport multimap

IF HAVE_QD:
    from decl cimport gso_mpz_dd, gso_mpz_qd, dd_t, qd_t
    from fplll cimport FT_DD, FT_QD

class EnumerationError(Exception):
    pass

class EvaluatorStrategy:
    """Strategies to update the enumeration radius and deal with multiple solutions.
    Possible values are:

    EvaluatorStrategy.
        BEST_N_SOLUTIONS
            Starting with the nr_solutions-th solution, every time a new solution is found
            the enumeration bound is updated to the length of the longest solution. If
            more than nr_solutions were found, the longest is dropped.
        OPPORTUNISTIC_N_SOLUTIONS
            Every time a solution is found, update the enumeration distance to the length
            of the solution. If more than nr_solutions were found, the longest is dropped.
        FIRST_N_SOLUTIONS
            The enumeration bound is not updated. As soon as nr_solutions are found,
            enumeration stops.
    """
    BEST_N_SOLUTIONS = EVALSTRATEGY_BEST_N_SOLUTIONS
    OPPORTUNISTIC_N_SOLUTIONS = EVALSTRATEGY_OPPORTUNISTIC_N_SOLUTIONS
    FIRST_N_SOLUTIONS = EVALSTRATEGY_FIRST_N_SOLUTIONS


cdef class Enumeration:
    def __init__(self, MatGSO M, nr_solutions=1, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS):
        """Create new enumeration object

        :param MatGSO M:      GSO matrix
        :param nr_solutions:  Number of solutions to be returned by enumeration
        :param strategy:      EvaluatorStrategy to use when finding new solutions

        """

        cdef MatGSO_c[Z_NR[mpz_t], FP_NR[double]]  *m_double
        IF HAVE_LONG_DOUBLE:
            cdef MatGSO_c[Z_NR[mpz_t], FP_NR[longdouble]] *m_ld
        cdef MatGSO_c[Z_NR[mpz_t], FP_NR[dpe_t]] *m_dpe
        IF HAVE_QD:
            cdef MatGSO_c[Z_NR[mpz_t], FP_NR[dd_t]] *m_dd
            cdef MatGSO_c[Z_NR[mpz_t], FP_NR[qd_t]] *m_qd
        cdef MatGSO_c[Z_NR[mpz_t], FP_NR[mpfr_t]]  *m_mpfr

        self.M = M

        if M._type == gso_mpz_d:
            m_double = M._core.mpz_d
            self._fe_core.d = new FastEvaluator_c[FP_NR[double]](nr_solutions,
                                                                      strategy,
                                                                      False)
            self._core.d = new Enumeration_c[FP_NR[double]](m_double[0], self._fe_core.d[0])
        elif M._type == gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                m_ld = M._core.mpz_ld
                self._fe_core.ld = new FastEvaluator_c[FP_NR[longdouble]](nr_solutions,
                                                                      strategy,
                                                                      False)
                self._core.ld = new Enumeration_c[FP_NR[longdouble]](m_ld[0], self._fe_core.ld[0])
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)
        elif M._type == gso_mpz_dpe:
            m_dpe = M._core.mpz_dpe
            self._fe_core.dpe = new FastEvaluator_c[FP_NR[dpe_t]](nr_solutions,
                                                                  strategy,
                                                                  False)
            self._core.dpe = new Enumeration_c[FP_NR[dpe_t]](m_dpe[0], self._fe_core.dpe[0])
        elif M._type == gso_mpz_mpfr:
            m_mpfr = M._core.mpz_mpfr
            self._fe_core.mpfr = new FastErrorBoundedEvaluator_c(M.d,
                                                                 M._core.mpz_mpfr.get_mu_matrix(),
                                                                 M._core.mpz_mpfr.get_r_matrix(),
                                                                 EVALMODE_SV,
                                                                 nr_solutions,
                                                                 strategy,
                                                                 False)
            self._core.mpfr = new Enumeration_c[FP_NR[mpfr_t]](m_mpfr[0], self._fe_core.mpfr[0])
        else:
            IF HAVE_QD:
                if M._type == gso_mpz_dd:
                    m_dd = M._core.mpz_dd
                    self._fe_core.dd = new FastEvaluator_c[FP_NR[dd_t]](nr_solutions,
                                                                           strategy,
                                                                           False)
                    self._core.dd = new Enumeration_c[FP_NR[dd_t]](m_dd[0], self._fe_core.dd[0])
                elif M._type == gso_mpz_qd:
                    m_qd = M._core.mpz_qd
                    self._fe_core.qd = new FastEvaluator_c[FP_NR[qd_t]](nr_solutions,
                                                                           strategy,
                                                                           False)
                    self._core.qd = new Enumeration_c[FP_NR[qd_t]](m_qd[0], self._fe_core.qd[0])
                else:
                    raise RuntimeError("MatGSO object '%s' has no core."%self)
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)

    def __dealloc__(self):
        if self.M._type == gso_mpz_d:
            del self._fe_core.d
            del self._core.d
        IF HAVE_LONG_DOUBLE:
            if self.M._type == gso_mpz_ld:
                del self._fe_core.ld
                del self._core.ld
        if self.M._type == gso_mpz_dpe:
            del self._fe_core.dpe
            del self._core.dpe
        IF HAVE_QD:
            if self.M._type == gso_mpz_dd:
                del self._fe_core.dd
                del self._core.dd
            if self.M._type == gso_mpz_qd:
                del self._fe_core.qd
                del self._core.qd
        if self.M._type == gso_mpz_mpfr:
            del self._fe_core.mpfr
            del self._core.mpfr

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
        :returns: list of pairs containing the solutions' coefficients and their lengths

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

        if self.M._type == gso_mpz_d:
            if target is not None:
                for it in target:
                    tmp.d = float(it)
                    target_coord_d.push_back(tmp.d)
            sig_on()
            self._core.d.enumerate(first, last, max_dist_d, max_dist_expo,
                                   target_coord_d, sub_tree_, pruning_, dual)
            sig_off()
            if not self._fe_core.d.size():
                raise EnumerationError("No vector found.")

            solutions_d = self._fe_core.d.begin()
            while solutions_d != self._fe_core.d.end():
                cur_dist = deref(solutions_d).first.get_d()
                cur_sol = []
                for j in range(deref(solutions_d).second.size()):
                    cur_sol.append(deref(solutions_d).second[j].get_d())
                solutions.append([tuple(cur_sol), cur_dist])
                inc(solutions_d)

        IF HAVE_LONG_DOUBLE:
            if self.M._type == gso_mpz_ld:
                if target is not None:
                    for it in target:
                        tmp.ld = float(it)
                        target_coord_ld.push_back(tmp.ld)
                sig_on()
                self._core.ld.enumerate(first, last, max_dist_ld, max_dist_expo,
                                        target_coord_ld, sub_tree_, pruning_, dual)
                sig_off()
                if not self._fe_core.ld.size():
                    raise EnumerationError("No vector found.")

                solutions_ld = self._fe_core.ld.begin()
                while solutions_ld != self._fe_core.ld.end():
                    cur_dist = deref(solutions_ld).first.get_d()
                    cur_sol = []
                    for j in range(deref(solutions_ld).second.size()):
                        cur_sol.append(deref(solutions_ld).second[j].get_d())
                    solutions.append([tuple(cur_sol), cur_dist])
                    inc(solutions_ld)

        if self.M._type == gso_mpz_dpe:
            if target is not None:
                for it in target:
                    tmp.dpe = float(it)
                    target_coord_dpe.push_back(tmp.dpe)
            sig_on()
            self._core.dpe.enumerate(first, last, max_dist_dpe, max_dist_expo,
                                     target_coord_dpe, sub_tree_, pruning_, dual)
            sig_off()
            if not self._fe_core.dpe.size():
                raise EnumerationError("No vector found.")

            solutions_dpe = self._fe_core.dpe.begin()
            while solutions_dpe != self._fe_core.dpe.end():
                cur_dist = deref(solutions_dpe).first.get_d()
                cur_sol = []
                for j in range(deref(solutions_dpe).second.size()):
                    cur_sol.append(deref(solutions_dpe).second[j].get_d())
                solutions.append([tuple(cur_sol), cur_dist])
                inc(solutions_dpe)

        IF HAVE_QD:
            if self.M._type == gso_mpz_dd:
                if target is not None:
                    for it in target:
                        tmp.dd = float(it)
                        target_coord_dd.push_back(tmp.dd)
                sig_on()
                self._core.dd.enumerate(first, last, max_dist_dd, max_dist_expo,
                                        target_coord_dd, sub_tree_, pruning_, dual)
                sig_off()
                if not self._fe_core.dd.size():
                    raise EnumerationError("No vector found.")

                solutions_dd = self._fe_core.dd.begin()
                while solutions_dd != self._fe_core.dd.end():
                    cur_dist = deref(solutions_dd).first.get_d()
                    cur_sol = []
                    for j in range(deref(solutions_dd).second.size()):
                        cur_sol.append(deref(solutions_dd).second[j].get_d())
                    solutions.append([tuple(cur_sol), cur_dist])
                    inc(solutions_dd)

            if self.M._type == gso_mpz_qd:
                if target is not None:
                    for it in target:
                        tmp.qd = float(it)
                        target_coord_qd.push_back(tmp.qd)
                sig_on()
                self._core.qd.enumerate(first, last, max_dist_qd, max_dist_expo,
                                        target_coord_qd, sub_tree_, pruning_, dual)
                sig_off()
                if not self._fe_core.qd.size():
                    raise EnumerationError("No vector found.")

                solutions_qd = self._fe_core.qd.begin()
                while solutions_qd != self._fe_core.qd.end():
                    cur_dist = deref(solutions_qd).first.get_d()
                    cur_sol = []
                    for j in range(deref(solutions_qd).second.size()):
                        cur_sol.append(deref(solutions_qd).second[j].get_d())
                    solutions.append([tuple(cur_sol), cur_dist])
                    inc(solutions_qd)

        if self.M._type == gso_mpz_mpfr:
            if target is not None:
                for it in target:
                    tmp.mpfr = float(it)
                    target_coord_mpfr.push_back(tmp.mpfr)
            sig_on()
            self._core.mpfr.enumerate(first, last, max_dist_mpfr, max_dist_expo,
                                      target_coord_mpfr, sub_tree_, pruning_, dual)
            sig_off()
            if not self._fe_core.mpfr.size():
                raise EnumerationError("No vector found.")

            solutions_mpfr = self._fe_core.mpfr.begin()
            while solutions_mpfr != self._fe_core.mpfr.end():
                cur_dist = deref(solutions_mpfr).first.get_d()
                cur_sol = []
                for j in range(deref(solutions_mpfr).second.size()):
                    cur_sol.append(deref(solutions_mpfr).second[j].get_d())
                solutions.append([tuple(cur_sol), cur_dist])
                inc(solutions_mpfr)

        return solutions

    def get_nodes(self):
        """Return number of visited nodes in last enumeration call.
        """
        if self.M._type == gso_mpz_d:
            return self._core.d.get_nodes()
        IF HAVE_LONG_DOUBLE:
            if self.M._type == gso_mpz_ld:
                return self._core.ld.get_nodes()
        if self.M._type == gso_mpz_dpe:
            return self._core.dpe.get_nodes()
        IF HAVE_QD:
            if self.M._type == gso_mpz_dd:
                return self._core.dd.get_nodes()
            if self.M._type == gso_mpz_qd:
                return self._core.qd.get_nodes()
        if self.M._type == gso_mpz_mpfr:
            return self._core.mpfr.get_nodes()
