# -*- coding: utf-8 -*-
import time

from fpylll import IntegerMatrix, GSO, LLL
from fpylll import BKZ
from fpylll import Enumeration as Enum
from fpylll import EnumerationError
from bkz_stats import BKZStats


class BKZReduction:
    def __init__(self, A):
        """FIXME! briefly describe function

        :param A:
        :returns:
        :rtype:

        """
        if not isinstance(A, IntegerMatrix):
            raise TypeError("Matrix must be IntegerMatrix but got type '%s'"%type(A))

        # run LLL first
        wrapper = LLL.Wrapper(A)
        wrapper()

        self.A = A
        self.m = GSO.Mat(A, flags=GSO.ROW_EXPO)
        self.lll_obj = LLL.Reduction(self.m)

    def __call__(self, param):
        """FIXME! briefly describe function

        :param block_size:
        :returns:
        :rtype:

        """
        stats = BKZStats(self, verbose=param.flags & BKZ.VERBOSE)

        if param.flags & BKZ.AUTO_ABORT:
            auto_abort = BKZ.AutoAbort(self.m, self.A.nrows)

        self.m.discover_all_rows()

        cputime_start = time.clock()

        i = 0
        while True:
            with stats.context("tour"):
                clean = self.tour(param, 0, self.A.nrows, stats)
            if clean or param.block_size >= self.A.nrows:
                break
            if (param.flags & BKZ.AUTO_ABORT) and auto_abort.test_abort():
                break
            if (param.flags & BKZ.MAX_LOOPS) and i >= param.max_loops:
                break
            if (param.flags & BKZ.MAX_TIME) and time.clock() - cputime_start >= param.max_time:
                break
            i += 1

    def tour(self, param, min_row, max_row, stats=None):
        """FIXME! briefly describe function

        :param block_size:
        :param min_row:
        :param max_row:
        :returns:
        :rtype:

        """
        clean = True
        for kappa in range(min_row, max_row-1):
            block_size = min(param.block_size, max_row - kappa)
            clean &= self.svp_reduction(kappa, param, block_size, stats)
            if stats:
                stats.log_clean_kappa(kappa, clean)
        return clean

    def svp_reduction(self, kappa, param, block_size, stats=None):
        """FIXME! briefly describe function

        :param kappa:
        :param block_size:
        :returns:
        :rtype:

        """
        clean = True

        if stats is None:
            stats = BKZStats(self)

        with stats.context("preproc"):
            self.lll_obj(0, kappa, kappa + block_size)
            if self.lll_obj.nswaps > 0:
                clean = False

            #  preprocessing
            if param.preprocessing:
                preproc = param.preprocessing
                auto_abort = BKZ.AutoAbort(self.m, kappa + block_size, kappa)
                cputime_start = time.clock()

                i = 0
                while True:
                    clean_inner = self.tour(preproc, kappa, kappa + block_size)
                    if clean_inner:
                        break
                    else:
                        clean = clean_inner
                    if auto_abort.test_abort():
                        break
                    if (preproc.flags & BKZ.MAX_LOOPS) and i >= preproc.max_loops:
                        break
                    if (preproc.flags & BKZ.MAX_TIME) and time.clock() - cputime_start >= preproc.max_time:
                        break
                    i += 1

        max_dist, expo = self.m.get_r_exp(kappa, kappa)
        delta_max_dist = self.lll_obj.delta * max_dist

        # TODO: Gaussian heuristic
        with stats.context("svp"):
            try:
                solution, max_dist = Enum.enumerate(self.m, max_dist, expo,
                                                    kappa, kappa + block_size,
                                                    param.pruning)
            except EnumerationError, msg:
                if param.flags & BKZ.GH_BND:
                    return clean
                else:
                    raise EnumerationError(msg)

        if max_dist >= delta_max_dist:
            return clean

        nonzero_vectors = len([x for x in solution if x])

        with stats.context("postproc"):
            if nonzero_vectors == 1:
                first_nonzero_vector = None
                for i in range(block_size):
                    if abs(solution[i]) == 1:
                        first_nonzero_vector = i
                        break

                self.m.move_row(kappa + first_nonzero_vector, kappa)
                self.lll_obj.size_reduction(kappa, kappa + 1)

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
