# -*- coding: utf-8 -*-

from __future__ import absolute_import
from fpylll.algorithms.simple_bkz import BKZReduction
from fpylll import Enumeration
import math
from functools import reduce


class DBKZReduction(BKZReduction):
    def bkz_loop(self, block_size, min_row, max_row):
        """FIXME! briefly describe function

        :param block_size:
        :param min_row:
        :param max_row:
        :returns:
        :rtype:

        """
        self.m.update_gso()
        clean = True
        for kappa in range(max_row - block_size, min_row - 1, -1):
            clean &= self.dsvp_reduction(kappa, block_size)
        clean &= BKZReduction.bkz_loop(self, block_size, min_row, max_row)
        return clean

    def euclid(self, pair1, pair2):
        """FIXME! briefly describe function

        :param pair1:
        :param pair2:
        :returns:
        :rtype:

        """
        row1, x1 = pair1
        row2, x2 = pair2
        if not x1:
            return pair2
        c = math.floor(x2/x1)
        self.m.row_addmul(row2, row1, -c)
        return self.euclid((row2, x2 - c*x1), pair1)

    def dsvp_reduction(self, kappa, block_size):
        """FIXME! briefly describe function

        :param kappa:
        :param block_size:
        :returns:
        :rtype:

        """
        clean = True

        self.lll_obj(0, kappa, kappa + block_size)
        if self.lll_obj.nswaps > 0:
            clean = False

        max_dist, expo = self.m.get_r_exp(kappa + block_size - 1, kappa + block_size - 1)
        max_dist = 1.0/max_dist
        expo *= -1.0
        delta_max_dist = self.lll_obj.delta * max_dist

        max_dist, solution = Enumeration(self.m).enumerate(kappa, kappa + block_size, max_dist, expo,
                                                           pruning=None, dual=True)[0]
        if max_dist >= delta_max_dist:
            return clean

        with self.m.row_ops(kappa, kappa+block_size):
            pairs = list(enumerate(solution, start=kappa))
            [self.m.negate_row(pair[0]) for pair in pairs if pair[1] < 0]
            pairs = map(lambda x: (x[0], abs(x[1])), pairs)
            # GCD should be tree based but for proof of concept implementation, this will do
            row, x = reduce(self.euclid, pairs)
            if x != 1:
                raise RuntimeError("Euclid failed!")
            self.m.move_row(row, kappa + block_size - 1)
        self.lll_obj(kappa, kappa, kappa + block_size)

        return False
