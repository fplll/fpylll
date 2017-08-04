# -*- coding: utf-8 -*-
"""
A variant of BKZ2 where tours apply svp_reduction selectively, namely, always choosing the block with 
the largest local slope.

..  moduleauthor:: Leo Ducas <ducas@cwi.nl>

..  note :: This module is purely experimental, and is only meant to verify some conjecture from 
Damien Sthele and Shi Bai. 

"""


from random import randint
from fpylll import BKZ, Enumeration, EnumerationError
from fpylll.algorithms.bkz2 import BKZReduction as BKZBase
from fpylll.algorithms.bkz_stats import dummy_tracer
from fpylll.util import gaussian_heuristic


class BKZReduction(BKZBase):

    def __init__(self, A):
        """Create new BKZ object.
        :param A: an integer matrix, a GSO object or an LLL object
        """
        BKZBase.__init__(self, A)
        self.lll_obj()

    def select_index(self, block_size, min_row, max_row):
        maxv = 1.
        r = self.M.r()
        for i in range(min_row, max_row-block_size-1):
            v = r[i] / gaussian_heuristic(r[i:i+block_size])
            if v > maxv:
                maxv, maxi = v, i
        return maxi

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        if max_row == -1:
            max_row = self.A.nrows
        for i in range(min_row, max_row-1):
            self.lll_obj.size_reduction()
            kappa = min_row + self.select_index(params.block_size,  min_row, max_row)

            block_size = min(params.block_size, max_row - kappa)
            self.svp_reduction(kappa, block_size, params, tracer)
        return False

    def __call__(self, params, min_row=0, max_row=-1):
        """Run the BKZ algorithm with parameters `param`.

        :param params: BKZ parameters
        :param min_row: start processing in this row
        :param max_row: stop processing in this row (exclusive)

        """
        tracer = BKZTreeTracer(self, verbosity=params.flags & BKZ.VERBOSE, start_clocks=True)

        with tracer.context("lll"):
            self.lll_obj()

        if not ((params.flags & BKZ.MAX_LOOPS) and params.max_loops > 0):
            raise ValueError("Selective BKZ requires flag BKZ.MAX_LOOPS and params.max_loops > 0")

        i = 0
        for i in xrange(params.max_loops):
            with tracer.context("tour", i):
                clean = self.tour(params, min_row, max_row, tracer)

        tracer.exit()
        self.trace = tracer.trace
        return clean
