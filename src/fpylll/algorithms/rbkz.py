# -*- coding: utf-8 -*-

from random import randint
from fpylll import BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZ1
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.algorithms.bkz_stats import DummyStats
from fpylll.util import gaussian_heuristic
from time import time
from fpylll.numpy import dump_r
from fpylll import prune


NODE_PER_SEC = 2.** 26
PREPROC_BLOCK_SIZE_INIT = 32
PREPROC_BLOCK_SIZE_INCR = 4


class BKZReduction(BKZ2):

    def __init__(self, A, recycling_pool_max_size=1):
        """Create new BKZ object.
        :param A: an integer matrix, a GSO object or an LLL object
        """
        BKZ2.__init__(self, A)
        self.recycling_pool_max_size = recycling_pool_max_size

    def recycled_svp_preprocessing(self, kappa, block_size, param, stats, preproc_block_size):

        prepar = param.__class__(block_size=preproc_block_size, strategies=param.strategies, flags=BKZ.GH_BND)
        clean = BKZ2.tour(self, prepar, kappa, kappa + block_size)

        return clean

    def tour(self, params, min_row=0, max_row=-1, stats=None):
        """One BKZ loop over all indices.
        :param params: BKZ parameters
        :param min_row: start index ≥ 0
        :param max_row: last index ≤ n
        :returns: ``True`` if no change was made and ``False`` otherwise
        """

        if max_row == -1:
            max_row = self.A.nrows

        clean = True

        for kappa in range(min_row, max_row-2):
            block_size = min(params.block_size, max_row - kappa)
            if block_size > 58:            
                clean &= self.recycled_svp_reduction(kappa, block_size, params, stats)
            else:
                clean &= self.svp_reduction(kappa, block_size, params, stats)
            if stats:
                stats.log_clean_kappa(kappa, clean)
        return clean

    def multi_insert(self, V, kappa, block_size, stats):
        d = self.M.d
        s = d
        l = len(V)
        print l,
        for w in V:
            self.M.create_row()
            with self.M.row_ops(s, s+1):
                for i in range(block_size):                    
                    self.M.row_addmul(s, kappa + i, w[i])
            s += 1

        for i in reversed(range(l)):
            self.M.move_row(s - 1, kappa)
        with stats.context("lll"):
            self.lll_obj(kappa, kappa, kappa + block_size + l)
        for i in range(l):
            self.M.move_row(kappa + block_size, s - 1)

        for i in range(l):
            self.M.remove_last_row()
        self.M.update_gso()
        return

    def recycled_svp_reduction(self, kappa, block_size, param, stats):
        """
        :param kappa:
        :param block_size:
        :param params:
        :param stats:
        """
        if stats is None:
            stats = DummyStats(self)

        self.M.update_gso()
        self.lll_obj.size_reduction(0, kappa+1)
        self.lll_obj(kappa, kappa, kappa + block_size)

        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize = 1.0, False
        print " - ",

        preproc_block_size = PREPROC_BLOCK_SIZE_INIT
        while remaining_probability > 1. - param.min_success_probability:
            preproc_block_size += PREPROC_BLOCK_SIZE_INCR

            start_preproc = time()
            with stats.context("preproc"):
                rec_clean = self.recycled_svp_preprocessing(kappa, block_size, param, stats, preproc_block_size)
            time_preproc = time() - start_preproc

            radius, expo = self.M.get_r_exp(kappa, kappa)

            if param.flags & BKZ.GH_BND:
                root_det = self.M.get_root_det(kappa, kappa+block_size)
                radius, expo = gaussian_heuristic(radius, expo, block_size, root_det, param.gh_factor)

            overhead = NODE_PER_SEC * time_preproc

            with stats.context("postproc"):
                self.M.update_gso()
                R = dump_r(self.M, kappa, block_size)
                # print R
                goal_proba = 1.01 * ((param.min_success_probability - 1)/remaining_probability + 1)
                pruning = prune(radius * 2**expo, overhead, goal_proba, [R],
                                descent_method="gradient", precision=53)

                print goal_proba, pruning.probability
            try:
                enum_obj = Enumeration(self.M, self.recycling_pool_max_size)
                aux_sols = []
                with stats.context("svp", E=enum_obj):
                    K = [x for x in pruning.coefficients]
                    radius *= 1.05
                    for i in range(5, preproc_block_size):
                        K[i] /= 1.05

                    solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, radius, expo,
                                                            pruning=K, aux_sols=aux_sols)
                    V = [v for (v, _) in aux_sols[:10]]
                    self.multi_insert(V, kappa, block_size, stats)

            except EnumerationError:
                print 0,
                pass

            remaining_probability *= (1 - pruning.probability)

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean


# def to_cannonical(A, v, kappa, block_size):
#     v = kappa*[0] + [x for x in v] + (A.nrows - (kappa + block_size)) * [0]
#     v = IntegerMatrix.from_iterable(1, A.nrows, map(lambda x: int(round(x)), v))
#     v = tuple((v*A)[0])
#     return v


# def multi_insert_from_cannonical(M, V, kappa, block_size):
#     d = M.d
#     s = d
#     l = len(V)
#     for v in V:
#         w = M.babai(v)
#         for i in range(kappa+block_size, d):
#             assert w[i] == 0
#         M.create_row()
#         with self.M.row_ops(s, s+1):
#             for i in range(kappa + block_size):
#                 self.M.row_addmul(s, i, w[i])
#         s += 1

#     for i in range(l).reversed():
#         self.M.move_row(kappa, d+i)

#     with stats.context("lll"):
#         self.lll_obj(kappa, kappa, kappa + block_size + 1)

#     for i in range(l):
#         self.M.move_row(kappa + block_size + i, s)

#     for i in range(l):
#         self.M.remove_last_row()