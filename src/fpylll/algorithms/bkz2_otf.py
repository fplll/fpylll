from random import randint
from fpylll import BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.tools.bkz_stats import dummy_tracer
from fpylll.util import gaussian_heuristic
from fpylll.fplll.pruner import prune
from fpylll.fplll.pruner import Pruning, PruningParams
from time import time

GRADIENT_BLOCKSIZE = 31
strenghts = range(10, 40, 2) + range(40, 120)
NPS = 60*[2.**29] + 5 * [2.**27] + 5 * [2.**26] + 1000 * [2.**25]


class BKZReduction(BKZBase):

    def __init__(self, A):
        """Create new BKZ object.
        :param A: an integer matrix, a GSO object or an LLL object
        """
        BKZBase.__init__(self, A)

    def get_pruning(self, kappa, block_size, params, target, preproc_cost, tracer=dummy_tracer):
        radius = self.M.get_r(kappa, kappa) * self.lll_obj.delta
        if block_size<30:
            return radius, PruningParams(4., ())

        r = [self.M.get_r(i, i) for i in range(kappa, kappa+block_size)]
        gh_radius = gaussian_heuristic(r)          
        if (params.flags & BKZ.GH_BND and block_size > 30):
            radius = min(radius, gh_radius * params.gh_factor)

        if not (block_size > GRADIENT_BLOCKSIZE):
            pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target, flags=0)
        else:
            while True:
                try: 
                    pruning = prune(radius, NPS[block_size] * preproc_cost, [r], target)
                    break
                except:
                    preproc_cost = 2*preproc_cost + .01

        return radius, pruning

    def svp_preprocessing(self, kappa, block_size, params, tracer=dummy_tracer, trials=0):
        clean = True

        BKZBase.svp_preprocessing(self, kappa, block_size, params, tracer=tracer)
        if trials == 0:
            return clean

        last_preproc = 2*(block_size/6) + trials + min(trials, 5)
        last_preproc = min(last_preproc, block_size - 10)
        preprocs = [last_preproc]

        while last_preproc > 30:
            last_preproc -= 8
            preprocs = [last_preproc] + preprocs

        for preproc in preprocs:
            prepar = params.__class__(block_size=preproc, flags=BKZ.BOUNDED_LLL)
            clean &= self.tour(prepar, kappa, kappa + block_size, tracer=tracer)

        return clean

    def svp_reduction(self, kappa, block_size, params, tracer=dummy_tracer):
        """
        :param kappa:
        :param block_size:
        :param params:
        :param tracer:
        """

        #if block_size < 30:
        #    return BKZBase.svp_reduction(self, kappa, block_size, params, tracer=tracer)

        self.lll_obj.size_reduction(0, kappa+1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)

        remaining_probability, rerandomize, trials = 1.0, False, 0

        while remaining_probability > 1. - params.min_success_probability:
            preproc_start = time()    
            with tracer.context("preprocessing"):
                if False: # ((trials%5)==4):
                    print "R", kappa, 
                    self.randomize_block(kappa+1, kappa+block_size, density=1, tracer=tracer)
                self.svp_preprocessing(kappa, block_size, params, tracer=tracer, trials=trials)
            preproc_cost = time() - preproc_start

            with tracer.context("pruner"):
                target = 1 - ((1. - params.min_success_probability) / remaining_probability)
                target =  min(target, .5)
                # target = params.min_success_probability
                radius, pruning = self.get_pruning(kappa, block_size, params, target*1.01, preproc_cost, tracer)

            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration",
                                    enum_obj=enum_obj,
                                    probability=pruning.expectation,
                                    full=block_size==params.block_size):
                    max_dist, solution = enum_obj.enumerate(kappa, kappa + block_size, radius, 0,
                                                            pruning=pruning.coefficients)[0]
                with tracer.context("postprocessing"):
                    rerandomize = True
                    self.svp_postprocessing(kappa, block_size, solution, tracer=tracer)

            except EnumerationError:
                rerandomize = False

            remaining_probability *= (1 - pruning.expectation)
            trials += 1

        self.lll_obj.size_reduction(0, kappa+1)
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2**(new_first_expo - old_first_expo)
        return clean