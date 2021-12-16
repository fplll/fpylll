# -*- coding: utf-8 -*-
"""
Block Korkine Zolotarev algorithm in Python.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>

This module reimplements fplll's BKZ algorithm in Python.  It has feature parity with the C++
implementation in fplll's core.  Additionally, this implementation collects some additional
statistics.  Hence, it should provide a good basis for implementing variants of this algorithm.
"""
from __future__ import absolute_import

try:
    from time import process_time  # Python 3
except ImportError:
    from time import clock as process_time  # Python 2
from fpylll import IntegerMatrix, GSO, LLL
from fpylll import BKZ
from fpylll import Enumeration
from fpylll import EnumerationError
from fpylll.util import adjust_radius_to_gh_bound
from fpylll.tools.bkz_stats import dummy_tracer, normalize_tracer, Tracer


class BKZReduction(object):
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
            raise TypeError("Matrix must be IntegerMatrix but got type '%s'" % type(A))

        if M is None and L is None:
            # run LLL first, but only if a matrix was passed
            LLL.reduction(A)

        self.A = A
        if M is None:
            self.M = GSO.Mat(A, flags=GSO.ROW_EXPO)
        else:
            self.M = M
        if L is None:
            self.lll_obj = LLL.Reduction(self.M, flags=LLL.DEFAULT)
        else:
            self.lll_obj = L

    def __call__(self, params, min_row=0, max_row=-1, tracer=False):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)
        :param tracer: see ``normalize_tracer`` for accepted values


        TESTS::

            >>> from fpylll import *
            >>> A = IntegerMatrix.random(60, "qary", k=30, q=127)
            >>> from fpylll.algorithms.bkz import BKZReduction
            >>> bkz = BKZReduction(A)
            >>> _ = bkz(BKZ.EasyParam(10), tracer=True); bkz.trace is None
            False
            >>> _ = bkz(BKZ.EasyParam(10), tracer=False); bkz.trace is None
            True

        """

        tracer = normalize_tracer(tracer)

        try:
            label = params["name"]
        except KeyError:
            label = "bkz"

        if not isinstance(tracer, Tracer):
            tracer = tracer(
                self,
                root_label=label,
                verbosity=params.flags & BKZ.VERBOSE,
                start_clocks=True,
                max_depth=2,
            )

        if params.flags & BKZ.AUTO_ABORT:
            auto_abort = BKZ.AutoAbort(self.M, self.A.nrows)

        cputime_start = process_time()

        with tracer.context("lll"):
            self.lll_obj()

        i = 0
        while True:
            with tracer.context("tour", i, dump_gso=params.flags & BKZ.DUMP_GSO):
                clean = self.tour(params, min_row, max_row, tracer)
            i += 1
            if clean or params.block_size >= self.A.nrows:
                break
            if (params.flags & BKZ.AUTO_ABORT) and auto_abort.test_abort():
                break
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break
            if (params.flags & BKZ.MAX_TIME) and process_time() - cputime_start >= params.max_time:
                break

        tracer.exit()
        try:
            self.trace = tracer.trace
        except AttributeError:
            self.trace = None
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

        for kappa in range(min_row, max_row - 1):
            block_size = min(params.block_size, max_row - kappa)
            clean &= self.svp_reduction(kappa, block_size, params, tracer)

        self.lll_obj.size_reduction(max(0, max_row - 1), max_row, max(0, max_row - 2))
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

    def svp_call(self, kappa, block_size, params, tracer=dummy_tracer):
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
            root_det = self.M.get_root_det(kappa, kappa + block_size)
            max_dist, expo = adjust_radius_to_gh_bound(
                max_dist, expo, block_size, root_det, params.gh_factor
            )

        try:
            enum_obj = Enumeration(self.M)
            with tracer.context("enumeration", enum_obj=enum_obj, probability=1.0):
                max_dist, solution = enum_obj.enumerate(kappa, kappa + block_size, max_dist, expo)[
                    0
                ]

        except EnumerationError as msg:
            if params.flags & BKZ.GH_BND:
                return None
            else:
                raise EnumerationError(msg)

        if max_dist >= delta_max_dist * (1 << expo):
            return None
        else:
            return solution

    def svp_postprocessing(self, kappa, block_size, solution, tracer=dummy_tracer):
        """Insert SVP solution into basis. Note that this does not run LLL; instead,
           it resolves the linear dependencies internally.

        :param solution: coordinates of an SVP solution
        :param kappa: current index
        :param block_size: block size
        :param tracer: object for maintaining statistics

        :returns: ``True`` if no change was made and ``False`` otherwise

        ..  note :: postprocessing does not necessarily leave the GSO in a safe state.  You may
            need to call ``update_gso()`` afterwards.
        """
        if solution is None:
            return True

        # d = self.M.d
        # self.M.create_row()

        # with self.M.row_ops(d, d+1):
        #     for i in range(block_size):
        #         self.M.row_addmul(d, kappa + i, solution[i])

        # self.M.move_row(d, kappa)
        # with tracer.context("lll"):
        #     self.lll_obj(kappa, kappa, kappa + block_size + 1)
        # self.M.move_row(kappa + block_size, d)
        # self.M.remove_last_row()

        j_nz = None

        for i in range(block_size)[::-1]:
            if abs(solution[i]) == 1:
                j_nz = i
                break

        if len([x for x in solution if x]) == 1:
            self.M.move_row(kappa + j_nz, kappa)

        elif j_nz is not None:
            with self.M.row_ops(kappa + j_nz, kappa + j_nz + 1):
                for i in range(block_size):
                    if solution[i] and i != j_nz:
                        self.M.row_addmul(kappa + j_nz, kappa + i, solution[j_nz] * solution[i])

            self.M.move_row(kappa + j_nz, kappa)

        else:
            solution = list(solution)

            for i in range(block_size):
                if solution[i] < 0:
                    solution[i] = -solution[i]
                    self.M.negate_row(kappa + i)

            with self.M.row_ops(kappa, kappa + block_size):
                offset = 1
                while offset < block_size:
                    k = block_size - 1
                    while k - offset >= 0:
                        if solution[k] or solution[k - offset]:
                            if solution[k] < solution[k - offset]:
                                solution[k], solution[k - offset] = (
                                    solution[k - offset],
                                    solution[k],
                                )
                                self.M.swap_rows(kappa + k - offset, kappa + k)

                            while solution[k - offset]:
                                while solution[k - offset] <= solution[k]:
                                    solution[k] = solution[k] - solution[k - offset]
                                    self.M.row_addmul(kappa + k - offset, kappa + k, 1)

                                solution[k], solution[k - offset] = (
                                    solution[k - offset],
                                    solution[k],
                                )
                                self.M.swap_rows(kappa + k - offset, kappa + k)
                        k -= 2 * offset
                    offset *= 2

            self.M.move_row(kappa + block_size - 1, kappa)

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

        self.lll_obj.size_reduction(0, kappa + 1)
        return clean
