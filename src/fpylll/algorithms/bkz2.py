# -*- coding: utf-8 -*-

from random import randint
from fpylll import LLL, GSO, BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.algorithms.bkz_stats import DummyStats
from fpylll.util import gaussian_heuristic


class BKZReduction(BKZBase):

    def __init__(self, A, gso_flags=GSO.ROW_EXPO, lll_flags=LLL.DEFAULT):
        """Create new BKZ object.

        :param A: matrix to work on
        :param gso_flags: flags to pass to GSO object
        :param lll_flags: flags to pass to LLL object

        """
        BKZBase.__init__(self, A, gso_flags=gso_flags, lll_flags=lll_flags)
        self.M.discover_all_rows()  # TODO: this belongs in __call__ (?)

    def get_pruning(self, kappa, block_size, param, stats=None):
        strategy = param.strategies[block_size]

        radius, re = self.M.get_r_exp(kappa, kappa)
        root_det = self.M.get_root_det(kappa, kappa + block_size)
        gh_radius, ge = gaussian_heuristic(radius, re, block_size, root_det, 1.0)
        return strategy.get_pruning(radius  * 2**re, gh_radius * 2**ge)

    def randomize_block(self, min_row, max_row, stats, density=0):
        """Randomize basis between from ``min_row`` and ``max_row`` (exclusive)

            1. permute rows

            2. apply lower triangular matrix with coefficients in -1,0,1

            3. LLL reduce result

        :param min_row: start in this row
        :param max_row: stop at this row (exclusive)
        :param stats: object for maintaining statistics
        :param density: number of non-zero coefficients in lower triangular transformation matrix
        """
        if max_row - min_row < 2:
            return  # there is nothing to do

        # 1. permute rows
        niter = 4 * (max_row-min_row)  # some guestimate
        with self.M.row_ops(min_row, max_row):
            for i in range(niter):
                b = a = randint(min_row, max_row-1)
                while b == a:
                    b = randint(min_row, max_row-1)
                self.M.move_row(b, a)

        # 2. triangular transformation matrix with coefficients in -1,0,1
        with self.M.row_ops(min_row, max_row):
            for a in range(min_row, max_row-2):
                for i in range(density):
                    b = randint(a+1, max_row-1)
                    s = randint(0, 1)
                    self.M.row_addmul(a, b, 2*s-1)

        return

    def svp_preprocessing(self, kappa, block_size, param, stats):
        clean = True

        clean &= BKZBase.svp_preprocessing(self, kappa, block_size, param, stats)

        for preproc in param.strategies[block_size].preprocessing_block_sizes:
            prepar = param.__class__(block_size=preproc, strategies=param.strategies, flags=BKZ.GH_BND)
            clean &= self.tour(prepar, kappa, kappa + block_size)

        return clean

    def svp_reduction(self, kappa, block_size, param, stats):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param stats:

        """
        if stats is None:
            stats = DummyStats(self)

        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize = 1.0, False

        while remaining_probability > 1. - param.min_success_probability:
            with stats.context("preproc"):
                if rerandomize:
                    self.randomize_block(kappa+1, kappa+block_size,
                                         density=param.rerandomization_density, stats=stats)
                self.svp_preprocessing(kappa, block_size, param, stats)

            radius, expo = self.M.get_r_exp(kappa, kappa)
            radius *= self.lll_obj.delta

            if param.flags & BKZ.GH_BND and block_size > 30:
                root_det = self.M.get_root_det(kappa, kappa + block_size)
                radius, expo = gaussian_heuristic(radius, expo, block_size, root_det, param.gh_factor)

            pruning = self.get_pruning(kappa, block_size, param, stats)

            try:
                enum_obj = Enumeration(self.M)
                with stats.context("svp", E=enum_obj):
                    solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, radius, expo,
                                                            pruning.coefficients)
                self.svp_postprocessing(kappa, block_size, solution, stats)
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            remaining_probability *= (1 - pruning.probability)

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean
