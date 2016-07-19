# -*- coding: utf-8 -*-

from random import randint
from fpylll import LLL, GSO, BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.algorithms.bkz_stats import DummyStats
from fpylll.util import get_root_det, compute_gaussian_heuristic


class BKZReduction(BKZBase):

    def __init__(self, A, gso_flags=GSO.ROW_EXPO, lll_flags=LLL.DEFAULT):
        """Create new BKZ object.

        :param A: matrix to work on
        :param gso_flags: flags to pass to GSO object
        :param lll_flags: flags to pass to LLL object

        """
        BKZBase.__init__(self, A, gso_flags=gso_flags, lll_flags=lll_flags)
        self.M.discover_all_rows()  # TODO: this belongs in __call__ (?)

    def get_pruning(self, kappa, block_size, param):
        strategy = param.strategies[block_size]

        root_det = get_root_det(self.M, kappa, kappa + block_size)
        radius, re = self.M.get_r_exp(kappa, kappa)
        gh_radius, ge = compute_gaussian_heuristic(block_size, root_det, 1.0)
        if gh_radius == 0.0:
            gh_radius, ge = radius, re

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
        niter = 3 * (max_row-min_row) + (max_row-min_row)**2 / 4  # some guestimate
        with self.M.row_ops(min_row, max_row):
            for i in xrange(niter):
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

        # 3. LLL reduce
        with stats.context("lll"):
            self.lll_obj(0, min_row, max_row)
        return

    def svp_preprocessing(self, kappa, block_size, param):
        clean = True

        for preproc in param.strategies[block_size].preprocessing_block_sizes:
            prepar = BKZ.Param(block_size=preproc, strategies=param.strategies)
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

        clean = True

        with stats.context("preproc"):
            with stats.context("lll"):
                self.lll_obj(kappa, kappa, kappa+block_size)
                if self.lll_obj.nswaps > 0:
                    clean = False

        remaining_probability, rerandomize = 1.0, False

        while remaining_probability > 1. - param.min_success_probability:
            with stats.context("preproc"):
                if rerandomize:
                    self.randomize_block(kappa+1, kappa+block_size,
                                         density=param.rerandomization_density, stats=stats)

                clean &= self.svp_preprocessing(kappa, block_size, param)

            radius, expo = self.M.get_r_exp(kappa, kappa)
            radius *= self.lll_obj.delta

            if param.flags & BKZ.GH_BND and block_size > 30:
                root_det = get_root_det(self.M, kappa, kappa + block_size)
                radius, expo = compute_gaussian_heuristic(block_size, root_det, param.gh_factor)

            pruning = self.get_pruning(kappa, block_size, param)

            try:
                E = Enumeration(self.M)
                with stats.context("svp", E=E):
                    solution, max_dist = E.enumerate(kappa, kappa + block_size, radius, expo, pruning.coefficients)
                clean &= self.svp_postprocessing(kappa, block_size, solution, stats)
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            self.M.update_gso() # HACK
            remaining_probability *= (1 - pruning.probability)

        return clean
