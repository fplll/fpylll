# -*- coding: utf-8 -*-
include "config.pxi"
include "cysignals/signals.pxi"


from libcpp.vector cimport vector
from gso cimport MatGSO
from fplll cimport Enumeration as Enumeration_c
from fplll cimport MatGSO as MatGSO_c
from fplll cimport Z_NR, FP_NR, mpz_t
from fplll cimport FastEvaluator
from fpylll cimport mpz_double, mpz_mpfr

class EnumerationError(Exception):
    pass

cdef class Enumeration:

    @staticmethod
    def enumerate(MatGSO M, max_dist, max_dist_expo, int first, int last, pruning, dual=False):
        """Run enumeration on `M`

        :param MatGSO M:       GSO matrix to run enumeration on
        :param max_dist:       length bound
        :param max_dist_expo:  exponent of length bound
        :param first:          first index
        :param last:           last index
        :param pruning:        pruning parameters
        :param dual:           run enumeration in the primal or dual lattice.
        :returns: solution, length

        """
        if M._type != mpz_double:
            raise NotImplementedError("Only mpz_double supported for now.")

        cdef MatGSO_c[Z_NR[mpz_t], FP_NR[double]] *gso = M._core.mpz_double

        cdef FastEvaluator[FP_NR[double]] evaluator

        cdef vector[double] pruning_
        cdef vector[FP_NR[double]] target_coord_
        cdef vector[FP_NR[double]] sub_tree_

        cdef int block_size = last-first

        if pruning is None:
            for i in range(block_size):
                pruning_.push_back(1)
        else:
            for i in range(block_size):
                pruning_.push_back(pruning[i])

        cdef double max_dist__ = max_dist
        cdef FP_NR[double] max_dist_ = max_dist__

        sig_on()
        Enumeration_c.enumerate[FP_NR[double]](gso[0], max_dist_, max_dist_expo, evaluator,
                                               target_coord_, sub_tree_,
                                               first, last, pruning_, dual)
        sig_off()

        if not evaluator.solCoord.size():
            raise EnumerationError("No vector found.")

        solution = []
        for i in range(evaluator.solCoord.size()):
            solution.append(evaluator.solCoord[i].get_d())

        max_dist = max_dist_.get_d()
        return tuple(solution), max_dist
