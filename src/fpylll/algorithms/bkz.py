# -*- coding: utf-8 -*-
"""
Block Korkine Zolotarev algorithm in Python.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>

This module reimplements fplll's BKZ algorithm in Python.  It has feature parity with the C++
implementation in fplll's core.  Additionally, this implementation collects some additional
statistics.  Hence, it should provide a good basis for implementing variants of this algorithm.
"""
from __future__ import absolute_import
import time
from fpylll import IntegerMatrix, GSO, LLL
from fpylll import BKZ
from fpylll import Enumeration
from fpylll import EnumerationError
from fpylll.util import gaussian_heuristic
from .bkz_stats import BKZTreeTracer, dummy_tracer


class BKZReduction:
    """
    An implementation of the BKZ algorithm in Python.

    This class has feature parity with the C++ implementation in fplll's core.  Additionally, this
    implementation collects some additional statistics.  Hence, it should provide a good basis for
    implementing variants of this algorithm.
    """
    def __init__(self, A):
        """Construct a new instance of the BKZ algorithm.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        if isinstance(A, GSO.Mat):
            L = None
            M = A
            A = M.B
        elif isinstance(A, LLL.Reduction):
            L = A
            M = L.M
            A = M.B
        elif isinstance(A, IntegerMatrix):
            L = None
            M = None
            A = A
        else:
            raise TypeError("Matrix must be IntegerMatrix but got type '%s'"%type(A))

        if M is None and L is None:
            # run LLL first, but only if a matrix was passed
            wrapper = LLL.Wrapper(A)
            wrapper()

        self.A = A
        if M is None:
            self.M = GSO.Mat(A, flags=GSO.ROW_EXPO)
        else:
            self.M = M
        if L is None:
            self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT)
        else:
            self.lll_obj = L

    def __call__(self, params, min_row=0, max_row=-1):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)

        """
        tracer = BKZTreeTracer(self, verbosity=params.flags & BKZ.VERBOSE)

        if params.flags & BKZ.AUTO_ABORT:
            auto_abort = BKZ.AutoAbort(self.M, self.A.nrows)

        cputime_start = time.clock()

        with tracer.context("lll"):
            self.lll_obj()

        i = 0
        while True:
            with tracer.context("tour", i):
                clean = self.tour(params, min_row, max_row, tracer)
            i += 1
            if clean or params.block_size >= self.A.nrows:
                break
            if (params.flags & BKZ.AUTO_ABORT) and auto_abort.test_abort():
                break
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break
            if (params.flags & BKZ.MAX_TIME) and time.clock() - cputime_start >= params.max_time:
                break

        self.trace = tracer.trace
        return clean

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
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
            clean &= self.svp_reduction(kappa, block_size, params, tracer)

        return clean

    def svp_preprocessing(self, kappa, block_size, params, tracer):
        """Perform preprocessing for calling the SVP oracle

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise

        .. note::

            ``block_size`` may be smaller than ``params.block_size`` for the last blocks.

        """
        clean = True

        lll_start = kappa if params.flags & BKZ.BOUNDED_LLL else 0
        with tracer.context("lll"):
            self.lll_obj(lll_start, lll_start, kappa + block_size)
            if self.lll_obj.nswaps > 0:
                clean = False

        return clean

    def svp_call(self, kappa, block_size, params, tracer=None):
        """Call SVP oracle

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: Coordinates of SVP solution or ``None`` if none was found.

        ..  note::

            ``block_size`` may be smaller than ``params.block_size`` for the last blocks.
        """
        max_dist, expo = self.M.get_r_exp(kappa, kappa)
        delta_max_dist = self.lll_obj.delta * max_dist

        if params.flags & BKZ.GH_BND:
            root_det = self.M.get_root_det(kappa, kappa+block_size)
            max_dist, expo = gaussian_heuristic(max_dist, expo, block_size, root_det, params.gh_factor)

        try:
            enum_obj = Enumeration(self.M)
            with tracer.context("enumeration", enum_obj=enum_obj, probability=1.0):
                solution, max_dist = enum_obj.enumerate(kappa, kappa + block_size, max_dist, expo)[0]

        except EnumerationError as msg:
            if params.flags & BKZ.GH_BND:
                return None
            else:
                raise EnumerationError(msg)

        if max_dist >= delta_max_dist * (1<<expo):
            return None
        else:
            return solution

    def svp_postprocessing(self, kappa, block_size, solution, tracer):
        """Insert SVP solution into basis and LLL reduce.

        :param solution: coordinates of an SVP solution
        :param kappa: current index
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        if solution is None:
            return True

        nonzero_vectors = len([x for x in solution if x])
        if nonzero_vectors == 1:
            first_nonzero_vector = None
            for i in range(block_size):
                if abs(solution[i]) == 1:
                    first_nonzero_vector = i
                    break

            self.M.move_row(kappa + first_nonzero_vector, kappa)
            with tracer.context("lll"):
                self.lll_obj.size_reduction(kappa, kappa + first_nonzero_vector + 1)

        else:
            d = self.M.d
            self.M.create_row()

            with self.M.row_ops(d, d+1):
                for i in range(block_size):
                    self.M.row_addmul(d, kappa + i, solution[i])

            self.M.move_row(d, kappa)
            with tracer.context("lll"):
                self.lll_obj(kappa, kappa, kappa + block_size + 1)
            self.M.move_row(kappa + block_size, d)
            self.M.remove_last_row()

        return False

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """Find shortest vector in projected lattice of dimension ``block_size`` and insert into
        current basis.

        :param kappa: current index
        :param params: BKZ parameters
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise
        """
        clean = True
        with tracer.context("preprocessing"):
            clean_pre = self.svp_preprocessing(kappa, block_size, params, tracer)
        clean &= clean_pre

        solution = self.svp_call(kappa, block_size, params, tracer)

        with tracer.context("postprocessing"):
            clean_post = self.svp_postprocessing(kappa, block_size, solution, tracer)
        clean &= clean_post

        return clean
