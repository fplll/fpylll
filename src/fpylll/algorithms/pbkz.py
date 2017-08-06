# -*- coding: utf-8 -*-
"""
Parallel BKZ reduction.

..  note :: This code only offers a noticeable performance improvement in block size 70 or so.
   Hence, it will simply call the sequential code for smaller block sizes.
"""

import os
import random
import multiprocessing

from fpylll import BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZ1
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.tools.bkz_stats import BKZTreeTracer, dummy_tracer
from fpylll.util import adjust_radius_to_gh_bound


class BKZReduction(BKZ2):
    """
    BKZ 2.0 with parallel SVP reduction.
    """
    def __init__(self, A, ncores=2):
        """Create new BKZ object.

        :param A: an integer matrix, a GSO object or an LLL object
        :param ncores: number of cores to use

        """
        self.ncores = ncores
        BKZ2.__init__(self, A)

    def svp_preprocessing(self, kappa, block_size, param, tracer=dummy_tracer):
        """
        Run sequential BKZ 2.0 preprocessing.

        :param kappa: current index
        :param block_size: block size
        :param params: BKZ parameters
        :param tracer: object for maintaining statistics

        """
        clean = True

        clean &= BKZ1.svp_preprocessing(self, kappa, block_size, param, tracer)

        for preproc in param.strategies[block_size].preprocessing_block_sizes:
            prepar = param.__class__(block_size=preproc, strategies=param.strategies, flags=BKZ.GH_BND)
            clean &= BKZ2.tour(self, prepar, kappa, kappa + block_size)

        return clean

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        """One BKZ tour over all indices.

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
            clean &= self.parallel_svp_reduction(kappa, block_size, params, tracer)

        return clean

    def parallel_svp_reduction_worker(self, kappa, block_size, params, rerandomize):
        """
        One SVP reduction, typically called in a worker process after forking.

        :param kappa: current index
        :param block_size: block size
        :param params: BKZ parameters
        :param tracer: object for maintaining statistics

        """
        # we create a new tracer object to report back our timings to the calling process
        tracer = BKZTreeTracer(self, verbosity=params.flags & BKZ.VERBOSE, root_label="svp")

        with tracer.context("preprocessing"):
            if rerandomize:
                with tracer.context("randomization"):
                    self.randomize_block(kappa+1, kappa+block_size,
                                         density=params.rerandomization_density, tracer=tracer)
            with tracer.context("reduction"):
                self.svp_preprocessing(kappa, block_size, params, tracer)

        radius, expo = self.M.get_r_exp(kappa, kappa)
        radius *= self.lll_obj.delta

        if params.flags & BKZ.GH_BND and block_size > 30:
            root_det = self.M.get_root_det(kappa, kappa + block_size)
            radius, expo = adjust_radius_to_gh_bound(radius, expo, block_size, root_det, params.gh_factor)

        pruning = self.get_pruning(kappa, block_size, params, tracer)

        try:
            enum_obj = Enumeration(self.M)
            with tracer.context("enumeration",
                                enum_obj=enum_obj,
                                probability=pruning.expectation,
                                full=block_size==params.block_size):
                max_dist, solution = enum_obj.enumerate(kappa, kappa + block_size, radius, expo,
                                                        pruning=pruning.coefficients)[0]
            with tracer.context("postprocessing"):
                # we translate our solution to the canonical basis because our basis is not
                # necessarily the basis of the calling process at this point
                solution = self.A.multiply_left(solution, start=kappa)

        except EnumerationError:
            solution, max_dist = None, None

        return solution, max_dist, tracer.trace, pruning.expectation

    def parallel_svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """
        SVP reduction attempts until the probability threshold is reached.

        :param kappa: current index
        :param block_size: block size
        :param params: BKZ parameters
        :param tracer: object for maintaining statistics

        .. note: This function uses for to parallelise.

        """
        # calling fork is expensive so we simply revert to the sequential code for small block sizes
        if block_size < 70:
            return self.svp_reduction(kappa, block_size, params, tracer=tracer)

        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize = 1.0, False

        while remaining_probability > 1. - params.min_success_probability:

            with tracer.context("lll"):
                self.lll_obj(0, 0, kappa + block_size)

            pipes = []
            for i in range(self.ncores):
                parent_connection, child_connection = multiprocessing.Pipe()
                pipes.append(parent_connection)

                pid = os.fork()
                if pid == 0:
                    random.seed(os.getpid()+random.randint(0, 1<<20))
                    ret = self.parallel_svp_reduction_worker(kappa, block_size, params, True if i>0 else rerandomize)
                    child_connection.send(ret)
                    os._exit(0)

            solutions, rerandomize = set(), True
            for i in range(self.ncores):
                solution, length, trace, probability = pipes[i].recv()
                remaining_probability *= (1 - probability)
                for child in trace.children:
                    tracer.current.child(child.label).merge(child)
                if solution:
                    rerandomize = False
                    solutions.add((solution, length))

            solutions = sorted(solutions, key=lambda x: x[1])
            for solution, length in solutions:
                with tracer.context("postprocessing"):
                    solution = self.M.babai(solution, start=kappa, dimension=block_size)
                    self.svp_postprocessing(kappa, block_size, solution, tracer)

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean
