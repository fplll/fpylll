# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"


from libcpp.vector cimport vector
from gso cimport MatGSO
from fplll cimport Enumeration as Enumeration_c
from fplll cimport FastEvaluator as FastEvaluator_c
from fplll cimport MatGSO as MatGSO_c
from fplll cimport Z_NR, FP_NR, mpz_t

from fplll cimport dpe_t
from fpylll.mpfr.mpfr cimport mpfr_t
from decl cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr, fp_nr_t
from fplll cimport FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR, FloatType

IF HAVE_QD:
    from fpylll.qd.qd cimport dd_real, qd_real
    from decl cimport mpz_dd, mpz_qd
    from fplll cimport FT_DD, FT_QD

class EnumerationError(Exception):
    pass

cdef class Enumeration:
    def __init__(self, MatGSO M):
        """Create new enumeration object

        :param MatGSO M: GSO matrix
        """

        cdef MatGSO_c[Z_NR[mpz_t], FP_NR[double]]  *m_double
        IF HAVE_LONG_DOUBLE:
            cdef MatGSO_c[Z_NR[mpz_t], FP_NR[longdouble]] *m_ld
        cdef MatGSO_c[Z_NR[mpz_t], FP_NR[dpe_t]] *m_dpe
        IF HAVE_QD:
            cdef MatGSO_c[Z_NR[mpz_t], FP_NR[dd_real]] *m_dd
            cdef MatGSO_c[Z_NR[mpz_t], FP_NR[qd_real]] *m_qd
        cdef MatGSO_c[Z_NR[mpz_t], FP_NR[mpfr_t]]  *m_mpfr

        self.M = M

        if M._type == mpz_double:
            m_double = M._core.mpz_double
            self._fe_core.double = new FastEvaluator_c[FP_NR[double]]()
            self._core.double = new Enumeration_c[FP_NR[double]](m_double[0], self._fe_core.double[0])
        elif M._type == mpz_ld:
            IF HAVE_LONG_DOUBLE:
                m_ld = M._core.mpz_ld
                self._fe_core.ld = new FastEvaluator_c[FP_NR[longdouble]]()
                self._core.ld = new Enumeration_c[FP_NR[longdouble]](m_ld[0], self._fe_core.ld[0])
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)
        elif M._type == mpz_dpe:
            m_dpe = M._core.mpz_dpe
            self._fe_core.dpe = new FastEvaluator_c[FP_NR[dpe_t]]()
            self._core.dpe = new Enumeration_c[FP_NR[dpe_t]](m_dpe[0], self._fe_core.dpe[0])
        elif M._type == mpz_mpfr:
            m_mpfr = M._core.mpz_mpfr
            self._fe_core.mpfr = new FastEvaluator_c[FP_NR[mpfr_t]]()
            self._core.mpfr = new Enumeration_c[FP_NR[mpfr_t]](m_mpfr[0], self._fe_core.mpfr[0])
        else:
            IF HAVE_QD:
                if M._type == mpz_dd:
                    m_dd = M._core.mpz_dd
                    self._fe_core.dd = new FastEvaluator_c[FP_NR[dd_real]]()
                    self._core.dd = new Enumeration_c[FP_NR[dd_real]](m_dd[0], self._fe_core.dd[0])
                elif M._type == mpz_qd:
                    m_qd = M._core.mpz_qd
                    self._fe_core.qd = new FastEvaluator_c[FP_NR[qd_real]]()
                    self._core.qd = new Enumeration_c[FP_NR[qd_real]](m_qd[0], self._fe_core.qd[0])
                else:
                    raise RuntimeError("MatGSO object '%s' has no core."%self)
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)

    def __dealloc__(self):
        if self.M._type == mpz_double:
            del self._fe_core.double
            del self._core.double
        IF HAVE_LONG_DOUBLE:
            if self.M._type == mpz_ld:
                del self._fe_core.ld
                del self._core.ld
        if self.M._type == mpz_dpe:
            del self._fe_core.dpe
            del self._core.dpe
        IF HAVE_QD:
            if self.M._type == mpz_dd:
                del self._fe_core.dd
                del self._core.dd
            if self.M._type == mpz_qd:
                del self._fe_core.qd
                del self._core.qd
        if self.M._type == mpz_mpfr:
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
        :returns: solution, length

        """
        cdef int block_size = last-first
        cdef fp_nr_t tmp

        cdef vector[FP_NR[double]] target_coord_d
        IF HAVE_LONG_DOUBLE:
            cdef vector[FP_NR[longdouble]] target_coord_ld
        cdef vector[FP_NR[dpe_t]] target_coord_dpe
        IF HAVE_QD:
            cdef vector[FP_NR[dd_real]] target_coord_dd
            cdef vector[FP_NR[qd_real]] target_coord_qd
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
        cdef FP_NR[double] max_dist_d = max_dist__
        IF HAVE_LONG_DOUBLE:
            cdef FP_NR[longdouble] max_dist_ld = max_dist__
        cdef FP_NR[dpe_t] max_dist_dpe = max_dist__
        IF HAVE_QD:
            cdef FP_NR[dd_real] max_dist_dd = max_dist__
            cdef FP_NR[qd_real] max_dist_qd = max_dist__
        cdef FP_NR[mpfr_t] max_dist_mpfr = max_dist__

        solution = []

        if self.M._type == mpz_double:
            if target is not None:
                for it in target:
                    tmp.double = float(it)
                    target_coord_d.push_back(tmp.double)
            sig_on()
            self._core.double.enumerate(first, last, max_dist_d, max_dist_expo,
                                        target_coord_d, sub_tree_, pruning_, dual)
            sig_off()
            if not self._fe_core.double.sol_coord.size():
                raise EnumerationError("No vector found.")

            for i in range(self._fe_core.double.sol_coord.size()):
                solution.append(self._fe_core.double.sol_coord[i].get_d())

            max_dist = max_dist_d.get_d()

        IF HAVE_LONG_DOUBLE:
            if self.M._type == mpz_ld:
                if target is not None:
                    for it in target:
                        tmp.ld = float(it)
                        target_coord_ld.push_back(tmp.ld)
                sig_on()
                self._core.ld.enumerate(first, last, max_dist_ld, max_dist_expo,
                                        target_coord_ld, sub_tree_, pruning_, dual)
                sig_off()
                if not self._fe_core.ld.sol_coord.size():
                    raise EnumerationError("No vector found.")

                for i in range(self._fe_core.ld.sol_coord.size()):
                    solution.append(self._fe_core.ld.sol_coord[i].get_d())

                max_dist = max_dist_ld.get_d()

        if self.M._type == mpz_dpe:
            if target is not None:
                for it in target:
                    tmp.dpe = float(it)
                    target_coord_dpe.push_back(tmp.dpe)
            sig_on()
            self._core.dpe.enumerate(first, last, max_dist_dpe, max_dist_expo,
                                     target_coord_dpe, sub_tree_, pruning_, dual)
            sig_off()
            if not self._fe_core.dpe.sol_coord.size():
                raise EnumerationError("No vector found.")

            for i in range(self._fe_core.dpe.sol_coord.size()):
                solution.append(self._fe_core.dpe.sol_coord[i].get_d())

            max_dist = max_dist_dpe.get_d()

        IF HAVE_QD:
            if self.M._type == mpz_dd:
                if target is not None:
                    for it in target:
                        tmp.dd = float(it)
                        target_coord_dd.push_back(tmp.dd)
                sig_on()
                self._core.dd.enumerate(first, last, max_dist_dd, max_dist_expo,
                                        target_coord_dd, sub_tree_, pruning_, dual)
                sig_off()
                if not self._fe_core.dd.sol_coord.size():
                    raise EnumerationError("No vector found.")

                for i in range(self._fe_core.dd.sol_coord.size()):
                    solution.append(self._fe_core.dd.sol_coord[i].get_d())

                max_dist = max_dist_dd.get_d()

            if self.M._type == mpz_qd:
                if target is not None:
                    for it in target:
                        tmp.qd = float(it)
                        target_coord_qd.push_back(tmp.qd)
                sig_on()
                self._core.qd.enumerate(first, last, max_dist_qd, max_dist_expo,
                                        target_coord_qd, sub_tree_, pruning_, dual)
                sig_off()
                if not self._fe_core.qd.sol_coord.size():
                    raise EnumerationError("No vector found.")

                for i in range(self._fe_core.qd.sol_coord.size()):
                    solution.append(self._fe_core.qd.sol_coord[i].get_d())

                max_dist = max_dist_qd.get_d()

        if self.M._type == mpz_mpfr:
            if target is not None:
                for it in target:
                    tmp.mpfr = float(it)
                    target_coord_mpfr.push_back(tmp.mpfr)
            sig_on()
            self._core.mpfr.enumerate(first, last, max_dist_mpfr, max_dist_expo,
                                      target_coord_mpfr, sub_tree_, pruning_, dual)
            sig_off()
            if not self._fe_core.mpfr.sol_coord.size():
                raise EnumerationError("No vector found.")

            for i in range(self._fe_core.mpfr.sol_coord.size()):
                solution.append(self._fe_core.mpfr.sol_coord[i].get_d())

            max_dist = max_dist_mpfr.get_d()

        return tuple(solution), max_dist

    def get_nodes(self):
        """Return number of visited nodes in last enumeration call.
        """
        if self.M._type == mpz_double:
            return self._core.double.get_nodes()
        IF HAVE_LONG_DOUBLE:
            if self.M._type == mpz_ld:
                return self._core.ld.get_nodes()
        if self.M._type == mpz_dpe:
            return self._core.dpe.get_nodes()
        IF HAVE_QD:
            if self.M._type == mpz_dd:
                return self._core.dd.get_nodes()
            if self.M._type == mpz_qd:
                return self._core.qd.get_nodes()
        if self.M._type == mpz_mpfr:
            return self._core.mpfr.get_nodes()
