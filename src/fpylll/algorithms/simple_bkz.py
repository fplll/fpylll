# -*- coding: utf-8 -*-
"""
A minimal implementation of the Block Korkine Zolotarev algorithm in Python.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

from __future__ import absolute_import
from fpylll import IntegerMatrix, GSO, LLL, BKZ
from fpylll import Enumeration


class BKZReduction:
    def __init__(self, A):
        """Construct a new BKZ reduction instance.

        :param A: Integer matrix to reduce.

        """
        if not isinstance(A, IntegerMatrix):
            raise TypeError("Matrix must be IntegerMatrix but got type '%s'"%type(A))

        # run LLL first
        wrapper = LLL.Wrapper(A)
        wrapper()

        self.A = A
        self.m = GSO.Mat(A, flags=GSO.ROW_EXPO)
        self.lll_obj = LLL.Reduction(self.m)

    def __call__(self, block_size):
        """Perform BKZ reduction with given``block_size``.

        Nothing is returned, the matrix ``A`` given during construction is modified in-place.

        :param block_size: an integer > 2

        """
        self.m.discover_all_rows()

        auto_abort = BKZ.AutoAbort(self.m, self.A.nrows)

        while True:
            clean = self.bkz_loop(block_size, 0, self.A.nrows)
            if clean:
                break
            if auto_abort.test_abort():
                break

    def bkz_loop(self, block_size, min_row, max_row):
        """Perform one BKZ loop, often also called a "BKZ tour".

        :param block_size: an integer > 2
        :param min_row: algorithm starts in this row (inclusive)
        :param max_row: algorithm stops at this row (exclusive)

        """
        clean = True
        for kappa in range(min_row, max_row-1):
            bs = min(block_size, max_row - kappa)
            clean &= self.svp_reduction(kappa, bs)
        return clean

    def svp_reduction(self, kappa, block_size):
        """Call the SVP oracle and insert found vector into basis.

        :param kappa: row index
        :param block_size: an integer > 2

        """
        clean = True

        self.lll_obj(0, kappa, kappa + block_size)
        if self.lll_obj.nswaps > 0:
            clean = False

        max_dist, expo = self.m.get_r_exp(kappa, kappa)
        delta_max_dist = self.lll_obj.delta * max_dist

        max_dist, solution = Enumeration(self.m).enumerate(kappa, kappa + block_size, max_dist, expo, pruning=None)[0]

        if max_dist >= delta_max_dist * (1<<expo):
            return clean

        nonzero_vectors = len([x for x in solution if x])

        if nonzero_vectors == 1:
            first_nonzero_vector = None
            for i in range(block_size):
                if abs(solution[i]) == 1:
                    first_nonzero_vector = i
                    break

            self.m.move_row(kappa + first_nonzero_vector, kappa)
            self.lll_obj.size_reduction(kappa, kappa + first_nonzero_vector + 1)

        else:
            d = self.m.d
            self.m.create_row()

            with self.m.row_ops(d, d+1):
                for i in range(block_size):
                    self.m.row_addmul(d, kappa + i, solution[i])

            self.m.move_row(d, kappa)
            self.lll_obj(kappa, kappa, kappa + block_size + 1)
            self.m.move_row(kappa + block_size, d)

            self.m.remove_last_row()

        return False
