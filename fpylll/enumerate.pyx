# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

include "interrupt/interrupt.pxi"

from libcpp.vector cimport vector
from gso cimport MatGSO
from fplll cimport Enumeration as Enumeration_c
from fplll cimport MatGSO as MatGSO_c
from fplll cimport Z_NR, FP_NR, mpz_t
from fplll cimport FastEvaluator
from fpylll cimport mpz_double, mpz_mpfr

cdef class Enumeration:

    @staticmethod
    def enumerate(MatGSO M, max_dist, max_dist_expo, first, last, pruning):
        """FIXME! briefly describe function

        :param MatGSO M:
        :param max_dist:
        :param max_dist_expo:
        :param first:
        :param last:
        :param pruning:
        :returns:
        :rtype:

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
                                               first, last, pruning_)
        sig_off()

        if not evaluator.solCoord.size():
            raise ArithmeticError("Enumeration failed.")

        solution = []
        for i in range(evaluator.solCoord.size()):
            solution.append(evaluator.solCoord[i].get_d())

        max_dist = max_dist_.get_d()
        return tuple(solution), max_dist
