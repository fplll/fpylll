# -*- coding: utf-8 -*-

from random import randint
from fpylll import BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz2 import BKZReduction as BKZBase
from fpylll.tools.bkz_stats import dummy_tracer
from fpylll.util import gaussian_heuristic
from fpylll.fplll.pruner import prune
from fpylll.fplll.pruner import Pruning
from time import time

GRADIENT_BLOCKSIZE = 31
SUBSOL_BLOCKSIZE = 41

NPS = 60*[2.**29] + 5 * [2.**27] + 5 * [2.**26] + 1000 * [2.**25]


class BKZReduction(BKZBase):

    def __init__(self, A):
        """Create new BKZ object.

        :param A: an integer matrix, a GSO object or an LLL object

        """
        BKZBase.__init__(self, A)

    def get_pruning(self, kappa, block_size, params, target, preproc_cost, tracer=dummy_tracer):
        radius = self.M.get_r(kappa, kappa) * self.lll_obj.delta

        r = [self.M.get_r(i, i) for i in range(kappa, kappa+block_size)]
        gh_radius = gaussian_heuristic(r)          
        if (params.flags & BKZ.GH_BND and block_size > 30):
            radius = min(radius, gh_radius * params.gh_factor)

        preproc_cost += .001
        if not (block_size > GRADIENT_BLOCKSIZE):
            pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target, flags=0)
        else: 
            try: 
                pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target)
            except:
                pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target, flags=0)

        return radius, pruning

    def svp_preprocessing(self, kappa, block_size, params, trials, tracer=dummy_tracer):
        clean = True

        lll_start = kappa if ((params.flags & BKZ.BOUNDED_LLL) or trials>0) else 0
        with tracer.context("lll"):
            self.lll_obj(lll_start, lll_start, kappa + block_size)
            if self.lll_obj.nswaps > 0:
                clean = False

        if trials < 3:
            return clean

        shift = trials - 3 

        last_preproc = 2*(block_size/5) + shift + min(shift, 5)
        last_preproc = min(last_preproc, block_size - 10)
        preprocs = [last_preproc]

        for preproc in preprocs:
            prepar = params.__class__(block_size=preproc, flags=BKZ.BOUNDED_LLL)
            clean &= self.tour(prepar, kappa, kappa + block_size, tracer=tracer)

        return clean

    def insert_sub_solutions(self, kappa, block_size, sub_solutions):
        M = self.M
        l = len(sub_solutions)
        n = M.d
        assert l < block_size

        for (a, vector) in sub_solutions:
            M.create_row()
            if len(vector)==0:      # No subsolution at this index. Leaving a 0 vector
                with M.row_ops(M.d-1, M.d):
                    M.row_addmul(M.d-1, kappa, 1)
                continue 
            with M.row_ops(M.d-1, M.d):
                for i in range(block_size):                    
                    M.row_addmul(M.d-1, kappa + i, vector[i])    

        for k in reversed(range(l)):
            M.move_row(M.d-1, kappa + k)

        self.lll_obj(kappa, kappa, kappa + block_size + l)

        for i in range(l):
            M.move_row(kappa + block_size, M.d-1)
            M.remove_last_row()

        return

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param tracer:

        """

        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize, trials = 1.0, False, 0

        sub_solutions = block_size > SUBSOL_BLOCKSIZE
        preproc_start = time()

        while remaining_probability > 1. - params.min_success_probability:
            with tracer.context("preprocessing"):
                self.svp_preprocessing(kappa, block_size, params, trials, tracer=tracer)
            preproc_cost = time() - preproc_start

            with tracer.context("pruner"):
                target = 1 - ((1. - params.min_success_probability) / remaining_probability)
                radius, pruning = self.get_pruning(kappa, block_size, params, target*1.01, 
                                                   preproc_cost, tracer)

            enum_obj = Enumeration(self.M, sub_solutions=sub_solutions)
            try:
                with tracer.context("enumeration",
                                    enum_obj=enum_obj,
                                    probability=pruning.expectation,
                                    full=block_size==params.block_size):
                    max_dist, solution = enum_obj.enumerate(kappa, kappa + block_size, radius, 0,
                                                            pruning=pruning.coefficients)[0]
                with tracer.context("postprocessing"):
                    preproc_start = time() # Include post_processing time as the part of the next pre_processing
                    if not sub_solutions:
                        self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)
                    if sub_solutions:
                        self.insert_sub_solutions(kappa, block_size, enum_obj.sub_solutions[:1+block_size/4])

            except EnumerationError:
                preproc_start = time()

            remaining_probability *= (1 - pruning.expectation)
            trials += 1

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean
